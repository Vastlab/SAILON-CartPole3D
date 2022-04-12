# UCCS TA 2 helper
import pdb
import numpy as np
import gym
from gym import make

import os
import random
import time
#from utils import rollout
import time
import pickle
import cv2
import PIL
import torch
import json
import argparse
from collections import OrderedDict
from functools import partial
from torch import Tensor
import torch.multiprocessing as mp
from my_lib import *
from statistics import mean
import scipy.stats as st

import gc
import random
import csv
import importlib.util
import math

from datetime import datetime, timedelta




try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class UCCSTA2():

    def __init__(self):

        # calibrated values for KL for cartpole wth one-step lookahead
        self.KL_threshold = 1
        self.KL_val = 0
        self.scoreforKL=20        # we only use the first sets of scores for KL because novels worlds close when balanced
        self.num_epochs = 200
        self.num_dims = 4
        self.num_blocks = None
        self.scalelargescores=20
        # takes a while for some randome starts to stabilise so don't reset too early as it
        # reduces world change sensitvity.  Effective min is 1 as need at least a prior state to get prediction.
        self.skipfirstNscores=1

        self.maxconsecutivefailthresh= 5  # if we see this many in a row we declare world changed as we never see even 3 in training 
        
        # we penalize for high failure rantes..  as  difference (faildiff*self.failscale) )
        self.failscale=6.0 #   How we scale failure fraction.. can be larger than one since its fractional differences and genaerally < .1 mostly < .05
        self.failfrac=.3  #Max fail fraction,  when above  this we start giving world-change probability for  failures

        #TB change from .06  for M30
        self.initprobscale=.25 #   we scale prob from initial state by this amount (scaled by 2**(consecuriteinit-2) and add world accumulator each time. No impacted by blend this balances risk from going of on non-novel worlds
        self.consecutiveinit=0   # if get consecutitve init failures we keep increasing scale
        self.consecutivedynamic=0   # if get consecutitve dynamic failures we keep increasing scale        

        # Large "control scores" often mean things are off, since we never know the exact model we reset when scores get
        # too large in hopes of  better ccotrol
        self.scoretoreset=1000

        #smoothed performance plot for dtection.. see perfscore.py for compuation.  Major changes in control mean these need updated
        self.perflist = []
        self.mean_perf = 0.8883502538071065
        self.stdev_perf = 0.11824239133691708
        self.PerfScale = 0.1    #How much do we weight Performacne KL prob.  make this small since it is slowly varying and added every episode. Small is  less sensitive (good for FP avoid, but yields slower detection). 

        self.consecutivesuccess=0
        self.consecutivefail=0
        self.maxconsecutivefail=0
        self.maxconsecutivesuccess=0        
        
        

        # TODO: change evm data dimensions
        if (self.num_dims == 4):
            self.mean_train = 0
            self.stdev_train = 0.0
            self.prob_scale = 2  # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
        else:
            self.mean_train = .198   #these are old values from Phase 1 2D cartpole..  for Pahse 2 3D we compute frm a training run.
            self.stdev_train = 0.051058052318592555
            self.mean_train = 0.002   #these guessted values for Phase 2 incase we get called without training
            self.stdev_train = 0.006
            self.prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.

        self.cnt = 0
        self.framecnt = 0
        self.saveframes = False
        self.saveprefix = random.randint(1,10000)
        
        self.worldchanged = 0
        self.worldchangedacc = 0
        self.blenduprate = .5           # fraction of new prob we use when blending up..  should be greater than  blenddown.  Note blendup uses max so never goes down.
        self.blenddownrate = .1         # fraction of new prob we use when blending down..  should be less than beld up rate.  No use of max
        
        self.failcnt = 0        
        self.worldchangeblend = 0
        # from WSU "train".. might need ot make this computed.
        #        self.mean_train=  0.10057711735799268
        #       self.stdev_train = 0.00016
        self.problist = []
        self.scorelist=[]
        self.maxprob = 0
        self.meanprob = 0
        self.noveltyindicator = None
        self.correctcnt=0
        self.rcorrectcnt=0        
        self.totalcnt=0        
        self.perf=0
        self.perm_search=0
        self.prev_action=0
        self.prev_state=None
        self.prev_predict=None        
        
        #        self.expected_backtwo = np.zeros(4)
        self.episode = 0
        self.trial = 0
        self.given = False
        self.statelist = []
        self.debug = False
        self.debug = True        
        self.debugstring = ""
        self.character=""
        self.trialchar=""        

        self.imax = self.imin =   self.imean = self.istd = self.ishape = self.iscale =  None
        self.dmax = self.dshape = self.dscale = self.dmean = self.dstd =   None        


        
        # Create prediction environment
        env_location = importlib.util.spec_from_file_location('CartPoleBulletEnv', \
                                                              'cartpolepp/UCCScart.py')
        env_class = importlib.util.module_from_spec(env_location)
        env_location.loader.exec_module(env_class)

        myconfig = dict()
        myconfig['start_zeroed_out'] = False
        

        # Package params here
        params = dict()
        params['seed'] = 0
        params['config'] = myconfig
#        params['path'] = "WSU-Portable-Generator/source/partial_env_generator/envs/cartpolepp"
        params['path'] = "./cartpolepp        "
        params['use_img'] = False
        params['use_gui'] = False

        self.uccscart = env_class.CartPoleBulletEnv(params)
        self.uccscart.path = "./cartpolepp"
        # with open('evm_config.json', 'r') as json_file:
        #     evm_config = json.loads(json_file.read())
        #     cover_threshold = evm_config['cover_threshold']
        #     distance_multiplier = evm_config['distance_multiplier']
        #     tail_size = evm_config['tail_size']
        #     distance_metric = evm_config['distance_metric']
        #     torch.backends.cudnn.benchmark = True
        #     args_evm = argparse.Namespace()
        #     args_evm.cover_threshold = [cover_threshold]
        #     args_evm.distance_multiplier = [distance_multiplier]
        #     args_evm.tailsize = [tail_size]
        #     args_evm.distance_metric = distance_metric
        #     args_evm.chunk_size = 200

        self.starttime = datetime.now()
        self.cumtime = self.starttime - datetime.now()
        return

    def reset(self, episode):
        self.problist = []
        self.scorelist=[]
        self.statelist = []
        self.given = False
        self.maxprob = 0
        self.meanprob = 0
        self.cnt = 0
        self.character=""
        self.debugstring=""        
        self.episode = episode
        self.worldchanged = 0
        self.uccscart.resetbase()
        self.uccscart.reset()
        self.uccscart.episode=episode
        
        if(episode ==0):  #reset things that we carry over between episodes withing the same trial
            self.worldchangedacc = 0
            self.failcnt = 0                    
            self.worldchangeblend = 0            
            self.consecutivefail=0
            self.perm_search=0
            self.trialchar=""
            



        # Take one step look ahead, return predicted environment and step

    # env should be a CartPoleBulletEnv
    def takeOneStep(self, state_given, env, pertub=False):
        observation = state_given
        action, next_action, expected_state = self.uccscart.get_best_action(observation,self.meanprob)
        # if doing well pertub it so we can better chance of detecting novelties
        '''
        ra = int(state_given[0] * 10000) % 4
        if (pertub and
                (ra == 0) and
                ((abs(state_given[0]) < .2)
                 and (abs(state_given[1]) < .25)
                 and (abs(state_given[2]) < .05)
                 and (abs(state_given[3]) < .1))):
            if (action == 1):
                action = 0
            else:
                action = 1'''
            # print("Flipped Action, state=",state_given)

        return action, expected_state


#####!!!!!##### Start INDEPNDENT CODE for EVT-
    def kullback_leibler(self, mu, sigma, m, s):
        '''
        Compute Kullback Leibler with Gaussian assumption of training data
        mu: mean of test batch
        sigm: standard deviation of test batch
        m: mean of all data in training data set
        s: standard deviation of all data in training data set
        return: KL ditance, non negative double precison float
        '''
        sigma = max(sigma, .0000001)
        s = max(s, .0000001)
        kl = np.log(s / sigma) + (((sigma ** 2) + ((mu - m) ** 2)) / (2 * (s ** 2))) - 0.5
        return kl

    #reversed wbl for maxim fitting
    def rwcdf(self,x,iloc,ishape,iscale):
        if(x-iloc< 0) : prob = 0        
        else: prob = 1-math.pow(math.exp(-(x-iloc)/iscale),ishape)
#        if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
        return prob

    #abs wbl for unsided fitting    
    def awcdf(self,x,iloc,ishape,iscale):
        prob = 1-math.pow(math.exp(-abs(x-iloc)/iscale),ishape)
#        if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
        return prob
    
    
    #regualr wbl for minimum fits
    def wcdf(self,x,iloc,ishape,iscale):
        if(iloc-x< 0) : prob = 0
        else: prob = 1-math.pow(math.exp(-(iloc-x)/iscale),ishape)
        #        if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
        return prob

    



#####!!!!!##### End Doimain Independent CODE for EVT-    
#####!!!!!##### Start Glue CODE for EVT-
    
    def format_istate_data(self,feature_vector):
        # Format data for use with evm
        #print(feature_vector)
        cur_state = []
        for i in feature_vector.keys():
            if i == 'time_stamp' or  i == 'image': continue             
            if i != 'blocks' and i != 'walls':
                #print(feature_vector[i])
                for j in feature_vector[i]:
                    #print(j)
                    cur_state.append(feature_vector[i][j])
            elif i == 'blocks':
                for block in feature_vector[i]:
                    for key in block.keys():
                        if key != 'id':
                            cur_state.append(block[key])
            elif i == 'walls':
                # Add in data for the walls
                for j in feature_vector[i]:
                    for k in j:
                        cur_state.append(k)

        return np.asarray(cur_state)


    def format_data(self, feature_vector):
        # Format data for use with evm
        state = []
        for i in feature_vector.keys():
#            if i != 'blocks' and i != 'time_stamp' and i != 'image' and i != 'ticks':
            if i == 'cart' or i == 'pole' :                
#                print(i,feature_vector[i])
                for j in feature_vector[i]:
                    state.append(feature_vector[i][j])
                #print(state)
        return np.asarray(state)

    def point_to_line_dist(self,istate,cpos, bpos, bvel):
        pdiff = np.subtract(cpos ,bpos)
        nval = np.linalg.norm(bvel)
        if(nval >0) :
            dist =  np.linalg.norm(np.cross(pdiff,bvel))/ nval
        else: dist = np.linalg.norm(pdiff)  # if vector direction (velocity) is 0 then distance is distance between the two points
        return dist

    def block_pos(self,istate,blocknum):
        return istate[13+blocknum*6:13+blocknum*6+2]

    def block_vel(self,istate,blocknum):
        return istate[13+blocknum*6+3:13+blocknum*6+5]

    def cart_pos(self,istate):
        return istate[0:2]

    def cart_vel(self,istate):
        return istate[3:5]    

    def pole_pos(self,istate):
        return istate[6:8]

    def pole_vel(self,istate):
        return istate[9:12]    


        

    # get probability differene froom initial state
    def istate_diff_EVT_prob(self,actual_state):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart Vel x" , " Cart Vel y" , " Cart Vel z ",  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole Vel x" , " Pole Vel y" , " Pole Vel z" , " Block x" , " Block y" , " Block x" ,  " Block Vel x" , " Block Vel y" , " Block Vel z" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]
        
        #load imin/imax from training..  with some extensions. From code some of these values don't seem plausable (blockx for example) but we saw them in training data.  maybe nic mixed up some parms/files but won't hurt too much fi we mis some
        #fitwblpy output for initial state data

        #max in min come directly for create_evm_data.py's output
        #if first time load up data.. 
        if(self.imax is None):
            #computed
            self.imax = np.array([2.986346e+00, 2.980742e+00, 0.000000e+00, 1.984800e-02,
                                  1.971800e-02, 0.000000e+00, 9.936000e-03, 9.933000e-03,
                                  9.938000e-03, 9.999980e-01, 2.725000e-02, 2.736900e-02,
                                  5.800000e-04, 4.090533e+00, 4.099369e+00, 9.122096e+00,
                                  9.842307e+00, 9.854365e+00, 9.848242e+00, 5.000000e+00,
                                  5.000000e+00, 0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00, 0.000000e+00,
                                  5.000000e+00, 5.000000e+00, 0.000000e+00, 5.000000e+00,
                                  5.000000e+00, 1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00, 1.000000e+01,
                                  5.000000e+00, 5.000000e+00, 1.000000e+01])
            #adjusted based on code.. 
            self.imax = np.array([3.0+00, 3.0+00, 0.000000e+00,  #cart pos
                                  2.e-02, 2.e-02, 0.000000e+00,    #cart vel
                                  1.0e-02, 1.0e-02, 1.0e-02, 1.0e-00,    #pole pos quat
                                  2.8e-02, 2.8e-02, 5.800000e-01,  # pole vel
                                  4.50+00, 4.50+00, 9.50,  # block pos
                                  11.5, 11.5, 11.5, #block vel.. they can speed up before first time we see them.. program limit is 10 but gravity can accelerate
                                  5.000000e+00, 5.000000e+00, 
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01])
            self.imin = np.array([-3.0+00, -3.0+00, 0.000000e+00,  #cart pos
                                  -2.e-02, -2.e-02, 0.000000e+00,    #cart vel
                                  -1.0e-02, -1.0e-02, -1.0e-02, 9.999970e-01,    #pole pos quat
                                  -2.8e-02, 2.8e-02, 5.800000e-04,  # pole vel
                                  -4.50533e+00, -4.500533e+00, 0.05533,  # block pos
                                  4.5, 4.5, 4.5, #block vel  based on  programmed values.. though as it can drop with gravity on first step which we we don't see
                                  5.000000e+00, 5.000000e+00, 
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  0.000000e+00, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01, 5.000000e+00, 5.000000e+00,
                                  1.000000e+01])


            initwbl = np.array([[[1.38261180e+00, 0.00000000e+00, 5.97191369e-03],
                                 [3.00475911e+00, 0.00000000e+00, 3.95621482e-03],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.87280624e+00, 0.00000000e+00, 1.63449012e-04],
                                 [1.46913032e+00, 0.00000000e+00, 8.67032946e-05],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.81339070e+00, 0.00000000e+00, 2.99451005e-05],
                                 [2.69843300e+00, 0.00000000e+00, 3.59685725e-05],
                                 [1.74449197e+00, 0.00000000e+00, 1.80368642e-05],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.59112566e+00, 0.00000000e+00, 1.29455765e-03],
                                 [1.51114138e+00, 0.00000000e+00, 1.24665112e-03],
                                 [1.74512625e+00, 0.00000000e+00, 4.33527749e-05],
                                 [1.71965636e+00, 0.00000000e+00, 1.14079390e-02],
                                 [1.83879399e+00, 0.00000000e+00, 1.70432620e-02],
                                 [1.11644671e+00, 0.00000000e+00, 1.99680658e-02],
                                 [1.39686903e+00, 0.00000000e+00, 1.24734419e-02],
                                 [1.84194195e+00, 0.00000000e+00, 8.72732910e-03],
                                 [1.81573511e+00, 0.00000000e+00, 1.00200122e-02],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                [[1.24454719e+03, 0.00000000e+00, 5.97799946e+00],
                                 [8.91362642e+02, 0.00000000e+00, 5.96469200e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.04886210e+03, 0.00000000e+00, 3.97617801e-02],
                                 [1.10240472e+03, 0.00000000e+00, 3.96653361e-02],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [2.76761926e+03, 0.00000000e+00, 1.98806715e-02],
                                 [3.97782525e+03, 0.00000000e+00, 1.98873591e-02],
                                 [5.46739493e+03, 0.00000000e+00, 1.98999663e-02],
                                 [3.70416373e+01, 0.00000000e+00, 1.27680956e-04],
                                 [1.13524913e+02, 0.00000000e+00, 5.56421371e-02],
                                 [1.26356287e+02, 0.00000000e+00, 5.48046877e-02],
                                 [4.17379637e+01, 0.00000000e+00, 1.20566608e-03],
                                 [2.87871488e+03, 0.00000000e+00, 8.19146145e+00],
                                 [2.88337138e+03, 0.00000000e+00, 8.24032840e+00],
                                 [9.10654044e+02, 0.00000000e+00, 8.26470640e+00],
                                 [5.26836236e+02, 0.00000000e+00, 1.97381003e+01],
                                 [9.18064763e+03, 0.00000000e+00, 1.97215311e+01],
                                 [3.78463739e+03, 0.00000000e+00, 1.96914248e+01],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])


            self.ishape = initwbl[0,:,0]
            self.iscale = initwbl[0,:,2]

        imax = self.imax            
        imin = self.imin
        ishape = self.ishape
        iscale = self.iscale        
            

        
        initprob=0  # assume nothing new in world

        cart_pos = [actual_state['cart']['x_position'],actual_state['cart']['y_position'],actual_state['cart']['z_position']]
        cart_pos = np.asarray(cart_pos)

        charactermin=1e-2
        
        istate = self.format_istate_data(actual_state)
        # do base state for cart(6)  and pole (7)   looking at position and velocity. 
        for j in range (13):
            if(abs(istate[j]) > imax[j]):
                probv =  self.awcdf(istate[j],imax[j],iscale[j],ishape[j]);
                initprob += probv
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" + str(dimname[j]) + " init too large  " + str(round(istate[j],3)) +" " + str(round(imax[j],3)) +" " + str(round(probv,3))

                
            if(abs(istate[j]) < imin[j]):
                probv =  self.awcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);
                if(probv>charactermin and len(self.character) < 256):
                    initprob += probv
                    self.character +=  "&" +  str(dimname[j]) + " init too small  " + str(round(istate[j],3)) +" " + str(round(imin[j],3)) +" " + str(round(probv,3))


       ## look for walls to be in bad position 

        wallstart= len(istate) - 24
        if(wallstart < 19):  # whould have at least 2 blocks
                probv =  1
                initprob += probv
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  " Too few blocks (Level 2 change) Len="+wallstart
        if(wallstart > 19+5*6):  # sould have at most 5 blocks
                probv =  1
                initprob += probv
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  " Too Many blocks (Level 2 change) Len="+wallstart



       ## look for blocks to be in bad position or to have bad velocity
                    
        k=13 # where block data begins 
        self.num_blocks = 0    
        for j in range (13,wallstart,1):
            if(abs(istate[j]) > imax[k]):
                probv =  self.awcdf(abs(istate[j]),imax[j],iscale[j],ishape[j]);
                initprob += probv
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  str(dimname[k])+ " init block too large " +  " " + str(round(istate[j],3)) +" " + str(round(imax[k],3)) +" " + str(round(probv,3))
            if(abs(istate[j]) < imin[k]):
                probv =  self.awcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);
#                probv=  (imin[k] - istate[j])/ (abs(istate[j]) + abs(imin[k]))
                initprob += probv
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  " " + str(dimname[k]) + " init block too small " +  " " + str(round(istate[j],3)) +" " + str(round(imin[k],3)) +" " + str(round(probv,3))
            k = k +1
            if(k==19):
                self.num_blocks += 1
                k=13;   #reset for next block
                
        self.character  += ";"


       ## look for blocks to heading at cart's initital position

        probv=0                    
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k
#            dist = self.point_to_line_dist(istate[0:3],istate[k+nb*6:k+nb*6+2],istate[k+nb*6+3:k+nb*6+5])
            dist = self.point_to_line_dist(istate,self.cart_pos(istate),
                                           self.block_pos(istate,nb),
                                           self.block_vel(istate,nb))
            if(dist < 1e-3): # should do wlb fit on this.. but for now just a hack
                probv = 1            
            elif(dist < .01): # should do wlb fit on this.. but for now just a hack
                probv = (.01-dist)/(.01-1e-3)
                probv = probv*probv   # square it so its a bit more concentrated and smoother                        
            initprob += probv
            if(probv>charactermin and len(self.character) < 256):
                self.character +=  "&" +   " M30 Char Block " + str(nb) + " on initial direction attacking cart " +" with prob " + str(probv)


       ## look for blocks motions that heading to  other blocks initital position
       
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k                
#                dist = self.point_to_line_dist(istate[k+nb2*6:k+nb2*6+2],istate[k+nb*6:k+nb*6+2],istate[k+nb*6+3:k+nb*6+5])
                dist = self.point_to_line_dist(istate,self.block_pos(istate,nb2),
                                               self.block_pos(istate,nb),
                                               self.block_vel(istate,nb))
                
                if(dist < 1e-3): # should do wlb fit on this.. but for now just a hack.  Note blocks frequently can randomly do this so don't consider it too much novelty
                    probv = .25            
                elif(dist < .01): # should do wlb fit on this.. but for now just a hack
                    probv = .25*(.01-dist)/(.01-1e-3)
                initprob += probv
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +   " M30 Char Block " + str(nb) + " on initial direction aiming at block" + str(nb2) +" with prob " + str(probv)

                
       ## look for blocks motions that are parallel/or anti-parallel 
       
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                #  Use only block direction (velocity) for parallel test -- cross product should be near 0.  But if there vector norm is 0 then no cross product. fit
                if(np.linalg.norm(self.block_vel(istate,nb2))>0 and np.linalg.norm(self.block_vel(istate,nb))>0):
                    dist  = np.linalg.norm(np.cross(self.block_vel(istate,nb2), self.block_vel(istate,nb)))
                    if(dist < 1e-3): # should do wlb fit on this.. but for now just a hack.. note can frequently happen accidently so not a full novelty prob change
                        probv = .25          
                    elif(dist < .01): # should do wlb fit on this.. but for now just a hack
                        probv = .25*(.01-dist)/(.01-1e-3)
                        probv = probv*probv   # square it so its a bit more concentrated and smoother                        
                    initprob += probv
                    if(probv>charactermin and len(self.character) < 256):
                        self.character +=  "&" +   " M30 Char Block motion " + str(nb) + " is (anti-) parallel to  block" + str(nb2) +" with prob " + str(probv)


        self.character  += ";"

        #        was limiting to one.. its a prob of novelty overall.. so 
        if(initprob >1):
            self.character += "Iprob clamped from" +  str(initprob)             
            initprob = 1



        if(initprob > 1e-4):
             self.consecutiveinit = min(self.consecutiveinit+ 1,self.maxconsecutivefailthresh+2)
             if(self.uccscart.tbdebuglevel>1):             
                 print("Initprob cnt char ", initprob, self.cnt,self.character)
        else:
            self.consecutiveinit =0
        return initprob

    


    # get probability differene from checking state difference from prediction
    def cstate_diff_EVT_prob(self,cdiff):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart Vel x" , " Cart Vel y" , " Cart Vel z" ,  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole Vel x" , " Pole Vel y" , " Pole Vel z" , " Block x" , " Block y" , " Block x" ,  " Block Vel x" , " Block Vel y" , " Block Vel z" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]

        if(self.dmax is None):        
        
            #load data from triningn
            self.dmax =    np.array([2.5664000e-02, 4.1890000e-02, 0.0000000e+00, #cart pos
                                     4.2490000e-02, 4.4770000e-02, 0.0000000e+00, #cart vel
                                     1.9012000e-02, 7.1500000e-03,8.4417000e-02, 2.3041000e-02,  #pole quat
                                     2.8452940e+00, 3.3939100e+00, 1.3531405e+01, #pole vel
                                     1.4438300e-01, 1.3333600e-01, 1.1041900e-01, #block pos
                                     7.2191570e+00, 6.6668060e+00, 5.5209520e+00])#block vel

            diffwbl = np.array([[[3.89207845e-01, 0.00000000e+00, 1.26112630e-02],
                                      [2.08246398e+01, 0.00000000e+00, 8.14071727e-02],
                                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                      [1.80495509e+00, 0.00000000e+00, 5.13721453e-01], #TB hand adjusted 
                                      [1.39238147e+00, 0.00000000e+00, 1.63304631e-01],#TB hand adjusted  
                                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                      [2.39054403e+00, 0.00000000e+00, 6.48172095e-03],
                                      [8.41732321e-01, 0.00000000e+00, 3.70890868e-03],
                                      [1.46458996e+00, 0.00000000e+00, 2.27123147e-02],
                                      [1.48986272e+00, 0.00000000e+00, 3.98654470e-02],
                                      [4.55443947e+00, 0.00000000e+00, 1.54452261e+00],
                                      [5.44053827e+00, 0.00000000e+00, 9.21315026e-01],
                                      [3.23239311e+00, 0.00000000e+00, 1.13444676e+00],
                                      [4.68533570e+00, 0.00000000e+00, 1.45315737e-02],
                                      [1.20077120e+00, 0.00000000e+00, 1.51368367e-02],
                                      [2.31710759e+00, 0.00000000e+00, 2.05984072e-02],
                                      [4.68573228e+00, 0.00000000e+00, 7.26578269e-01],
                                      [1.20072688e+00, 0.00000000e+00, 7.56841161e-01],
                                      [2.31709025e+00, 0.00000000e+00, 1.02992254e+00],
                                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                     [[1.35507369e+01, 0.00000000e+00, 5.81425563e-02],
                                      [8.43288360e+02, 0.00000000e+00, 8.20148859e-02],
                                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                      [9.21130389e-01, 0.00000000e+00, 4.34356316e-03],
                                      [1.42393753e+00, 0.00000000e+00, 8.87714776e-03],
                                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                      [5.64946247e+00, 0.00000000e+00, 5.85444651e-02],
                                      [1.11560747e+01, 0.00000000e+00, 2.70439053e-02],
                                      [9.89439561e+00, 0.00000000e+00, 1.63992817e-01],
                                      [2.69850797e+01, 0.00000000e+00, 1.51578428e-01],
                                      [1.03975272e+02, 0.00000000e+00, 3.13070568e+00],
                                      [2.80463669e+01, 0.00000000e+00, 3.75921232e+00],
                                      [1.54359124e+01, 0.00000000e+00, 2.21769648e+01],
                                      [2.18464037e+01, 0.00000000e+00, 3.00350669e-01],
                                      [8.55836582e+01, 0.00000000e+00, 2.92873593e-01],
                                      [3.52209971e+01, 0.00000000e+00, 2.62699935e-01],
                                      [2.18463402e+01, 0.00000000e+00, 1.50175196e+01],
                                      [8.55804857e+01, 0.00000000e+00, 1.46436707e+01],
                                      [3.52198475e+01, 0.00000000e+00, 1.31349703e+01],
                                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])            

            self.dshape = diffwbl[0,:,0]
            self.dscale = diffwbl[0,:,2]        
        imax = self.dmax;
        imin = -imax
        ishape = self.dshape
        iscale = self.dscale

        #else  just initize locals

#        if(cdiff[0] > .01):
#            pdb.set_trace()
        
        if(self.cnt < self.skipfirstNscores): return 0;    #need to be far enough along to get good prediciton

        prob=0  #where we accumualte probability
        charactermin=1e-2
        istate = cdiff
        # do base state for cart(6)  and pole (7) 
        for j in range (13):        
            if(istate[j] > imax[j]):
#                probv =  self.awcdf(istate[j],imax[j],iscale[j],ishape[j]);
                probv =  self.awcdf(abs(istate[j]),imax[j],iscale[j],ishape[j])                
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  dimname[j] + " diff too large prob " + " " + str(round(probv,5)) + " s/l " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
                prob += probv
            elif(istate[j] < imin[j]):
                probv =  self.awcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);                
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  dimname[j] + " diff too small prob " + " " + str(round(probv,5)) + "  s/l " + str(round(istate[j],5)) +  " " + str(round(imin[j],5))
                prob += probv
        

        #no walls in dtate diff just looping over blocks
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,len(istate),1):
            if(istate[j] > imax[k]):
                probv =  self.awcdf(abs(istate[j]),imax[j],iscale[j],ishape[j]);                
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  " " + str(dimname[k]) + " diff too large, prob " + " "
                    + str(round(probv,5)) + "  s/l " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
                    prob += probv
            elif(istate[j] < imin[k]):
                probv =  self.awcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);                
                if(probv>charactermin and len(self.character) < 256):
                    self.character +=  "&" +  " " + str(dimname[k]) + " diff too small, prob " + " "
                    + str(round(probv,5)) + " s/l " + str(round(istate[j],5)) +  " " + str(round(imin[j],5))
                prob += probv
                k = k +1
                if(k==19): k=13;   #reset for next block
        self.character  += ";"                

        if(prob > 1e-2):
            if(prob > 1): prob = 1
            self.consecutivedynamic = min(self.consecutivedynamic+1,self.maxconsecutivefailthresh+2)
        else:
            self.consecutivedynamic =0


#        if(prob > charactermin): print("DProb @step/prob/char =",  self.cnt, prob, self.character)
#        pdb.set_trace()
        return prob




        # get probability differene froom initial state
    def istate_diff_G_prob(self,actual_state):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,
                 " Cart Vel x" , " Cart Vel y" , " Cart Vel z ",
                 " Pole x" , " pole y" , " Pole z" ," Pole w" ,
                 " Pole Vel x" , " Pole Vel y" , " Pole Vel z" ,
                 " Block x" , " Block y" , " Block x" ,
                 " Block Vel x" , " Block Vel y" , " Block Vel z" ,
                 " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]
        
        #load mean/std from training..  
        #if first time load up data.. 
        if(self.imean is None):
#            pdb.set_trace()
            #abuse tha terms since we are tryng to share code with  EVT version
            self.imean = np.array([-3.88847920e-02,  1.79388544e-02,  0.00000000e+00, -8.32127574e-05,
                                5.61661668e-05,  0.00000000e+00, -1.52076385e-04, -1.22921416e-05,
                                9.18328334e-05,  9.99950191e-01, -2.47030594e-05,  2.25482903e-04,
                                3.81943611e-06, -3.34521356e-03, -3.63842781e-02,  5.03422818e+00,
                                -1.33613935e-02, -3.96377125e-02,  3.47097780e-03, -5.00000000e+00,
                                -5.00000000e+00,  0.00000000e+00,  5.00000000e+00, -5.00000000e+00,
                                0.00000000e+00,  5.00000000e+00,  5.00000000e+00,  0.00000000e+00,
                                -5.00000000e+00,  5.00000000e+00,  0.00000000e+00, -5.00000000e+00,
                                -5.00000000e+00,  1.00000000e+01,  5.00000000e+00, -5.00000000e+00,
                                1.00000000e+01,  5.00000000e+00,  5.00000000e+00,  1.00000000e+01,
                                -5.00000000e+00,  5.00000000e+00,  1.00000000e+01])


            #adjusted based on code.. 
            self.istd = np.array([1.70842105e+00, 1.72660247e+00, 1.00000000e-06, 1.14153952e-02,
                                  1.14772434e-02, 1.00000000e-06, 5.79354090e-03, 5.75740406e-03,
                                  5.73292428e-03, 2.60735246e-05, 1.21789231e-02, 1.21918190e-02,
                                  1.96198184e-04, 2.32131483e+00, 2.33097704e+00, 2.29533051e+00,
                                  7.97934966e+00, 7.97332509e+00, 8.02280604e+00, 1.00000000e-06,
                                  1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                                  1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                                  1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                                  1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                                  1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                                  1.00000000e-06, 1.00000000e-06, 1.00000000e-06])



        imean = self.imean            
        istd = self.istd
        
        initprob=0  # assume nothing new in world

        cart_pos = [actual_state['cart']['x_position'],actual_state['cart']['y_position'],actual_state['cart']['z_position']]
        cart_pos = np.asarray(cart_pos)

        charactermin=1e-2
        
        istate = self.format_istate_data(actual_state)
        # do base state for cart(6)  and pole (7) 
        for j in range (13):
            probv =  self.gcdf(istate[j],imean[j],istd[j]);
            if(probv>charactermin and len(self.character) < 256):
#                initprob += probv
                initprob = max(initprob,probv)                
                self.character +=  "&" +   str(dimname[j]) + " init out of range  " + str(round(istate[j],3)) +" " + str(round(imean[j],3)) +" "  + str(round(istd[j],3)) +" " + str(round(probv,3))

        wallstart= len(istate) - 24                    
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,wallstart,1):
            if("Wall"  in str(dimname[j])): break
            probv =  self.gcdf(istate[j],imean[j],istd[j]);
            if(probv>charactermin and len(self.character) < 256):
#                initprob += probv
                initprob = max(initprob,probv)                
                self.character +=  "&" +   str(dimname[j]) + " block init out of range  " + str(round(istate[j],3)) +" " + str(round(imean[j],3)) +" "  + str(round(istd[j],3)) +" " + str(round(probv,3))
            k = k +1
            if(k==19): k=13;   #reset for next block
        self.character  += ";"

        if(initprob >1): initprob = 1

        if(initprob > 1e-4):
             self.consecutiveinit += 1
             if(self.uccscart.tbdebuglevel>1):             
                 print("Initprob cnt char ", initprob, self.cnt,self.character)
        else:
            self.consecutiveinit =0
        return initprob

    


    # get probability differene froom continuing state difference
    def cstate_diff_G_prob(self,cdiff):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart Vel x" , " Cart Vel y" , " Cart Vel z" ,  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole Vel x" , " Pole Vel y" , " Pole Vel z" , " Block x" , " Block y" , " Block x" ,  " Block Vel x" , " Block Vel y" , " Block Vel z" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]

        if(self.dmean is None):        
        
            #load data from triningn
            self.dmean =    np.array([-3.13321142e-07,  2.76174424e-05,  0.00000000e+00,
                                      -3.22495249e-05,-4.54427372e-05,  0.00000000e+00,
                                      -1.96683762e-06,  4.17344353e-06,-3.58822868e-06,  1.24106557e-04,
                                      -1.90599960e-03,  6.85632951e-04, 2.24843167e-03,
                                      -2.08509736e-06, -1.15790544e-05,  9.71314184e-05,
                                      -1.04261269e-04, -5.78904334e-04,  4.85657617e-03])
            self.dstd =    np.array([-3.13321142e-07,  2.76174424e-05,  1.00000000e-06,
                                     -3.22495249e-05, -4.54427372e-05,  1.00000000e-06,
                                     -1.96683762e-06,  4.17344353e-06,-3.58822868e-06,  1.24106557e-04,
                                     -1.90599960e-03,  6.85632951e-04, 2.24843167e-03,
                                     -2.08509736e-06, -1.15790544e-05,  9.71314184e-05,
                                     -1.04261269e-04, -5.78904334e-04,  4.85657617e-03])


        imean = self.dmean;
        istd = self.dstd;

        
        if(self.cnt < self.skipfirstNscores): return 0;    #need to be far enough along to get good prediciton

        prob=0  #where we accumualte probability
        charactermin=1e-2
        istate = cdiff
        # do base state for cart(6)  and pole (7) 
        for j in range (13):        
            probv =  self.gcdf(istate[j],imean[j],istd[j]);
            if(probv>charactermin and len(self.character) < 256):
#                prob += probv
                prob = max(prob,probv)                
                self.character +=  "&" +   str(dimname[j]) + " dyn out of range  " + str(round(istate[j],6)) +" " + str(round(imean[j],6)) +" "  + str(round(istd[j],6)) +" " + str(round(probv,3))

        

        #no walls in dtate diff just looping over blocks
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,len(istate),1):
            if("Wall"  in str(dimname[j])): break
            probv =  self.gcdf(istate[j],imean[j],istd[j]);
            if(probv>charactermin and len(self.character) < 256):
                prob += probv
                prob = max(prob,probv)                
                self.character +=  "&" +   str(dimname[j]) + " dyn out of range  " + str(round(istate[j],6)) +" " + str(round(imean[j],6)) +" "  + str(round(istd[j],6)) +" " + str(round(probv,3))
            k = k +1
            if(k==19): k=13;   #reset for next block
        self.character  += ";"                

        if(prob > 1e-2):
            if(prob > 1): prob = 1
            self.consecutivedynamic += 1
        else:
            self.consecutivedynamic =0


#        if(prob > charactermin): print("DProb @step/prob/char =",  self.cnt, prob, self.character)
#        pdb.set_trace()
        return prob


    


    def world_change_prob(self,settrain=False):

        # don't let first episodes  impact world change.. need stabilsied scores/probabilites
        if(self.episode<1):
            self.worldchangedacc = 0
            self.worldchangeblend = 0
            return self.worldchangedacc            
        
        
        mlength = len(self.problist)
        mlength = min(self.scoreforKL,mlength)
        # we look at the larger of the begging or end of list.. world changes most obvious at the ends. 

        previous_wc = self.worldchangedacc

        window_width=11
        if (len(self.perflist) >2*self.scoreforKL):   #look at list of performacne to see if its deviation from training is so that is.. skip more since it needs to be stable for window smoothing+ mean/variance computaiton
            #get smoothed performance 
            cumsum_vec = np.cumsum(np.insert(self.perflist, 0, 0))
            smoothed = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
            pmu = np.mean(smoothed[:-self.scoreforKL])  # we skip first/iniiprob... it is used elsehwere. 
            psigma = np.std(smoothed[:-self.scoreforKL])
            
            # if(pmu > self.mean_perf):     #if we want only  KL for those what are worse.. 
            #     PerfKL = 0 
            # else:
            PerfKL = self.kullback_leibler(pmu, psigma, self.mean_perf, self.stdev_perf)

            # If there is still too much variation (too many FP) in the variance in the small window so we use stdev and just new mean this allows smaller (faster) window for detection. 
            # PerfKL = self.kullback_leibler(pmu, self.stdev_perf, self.mean_perf, self.stdev_perf)
        else:
            PerfKL = 0

        
        

        if(mlength > 1) :
            mu = np.mean(self.problist[0:mlength-1])
            sigma = np.std(self.problist[0:mlength-1])
        else:
            mu = sigma = 0
            self.debugstring = '   ***Zero Lenth World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,5),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
            print(self.debugstring)
            
            self.worldchanged = self.worldchangedacc
            return self.worldchangedacc;
       
        if(settrain):
           self.mean_train = mu;
           self.stdev_train = sigma;
#           print("Set  world change train mu and sigma", mu, sigma)
           self.worldchanged = 0
           return 0;
        if( self.mean_train == 0):
           self.mean_train = 0.002   #these guessted values for Phase 2 incase we get called without training
           self.stdev_train = 0.006
           self.prob_scale = 2  # probably do need to scale but not tested sufficiently to see what it needs.

        self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)                
        self.debugstring = '   ***Short World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
            round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
            round(self.stdev_train,3) ,round(self.KL_val,5), round(self.KL_threshold,5), "\n", [round(num,2) for num in self.scorelist])
        if (self.debug):
            print(self.debugstring)
           

        if (len(self.problist) < 198):   #for real work
            self.consecutivesuccess=0            
            self.failcnt += 1
            if( self.consecutivefail >0):
                self.consecutivefail = min(self.consecutivefail+1, self.maxconsecutivefailthresh+2)
                if(self.consecutivefail > self.maxconsecutivefail):
                    self.maxconsecutivefail = self.consecutivefail
                    if(self.maxconsecutivefail > self.maxconsecutivefailthresh):
                        self.worldchangedacc = 1
                        self.character +=  "&" +  "#####? Uncontrollable world -- too many consecutive failures.  Guessing actions were remapped/perturbed but will take a while to confirm ##### "                         
                        
            else: self.consecutivefail=1

            self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)                
            self.debugstring = '   ***Short World Change Acc={}, Failcnt= {} Prob ={},,mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,3),self.failcnt, [round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
                round(self.stdev_train,3) ,round(self.KL_val,5), round(self.KL_threshold,5), "\n", [round(num,2) for num in self.scorelist])
            if (self.debug):
                print(self.debugstring)
        else:
            self.consecutivefail=0
            self.consecutivesuccess += 1
            if(self.consecutivesuccess > self.maxconsecutivesuccess):
                self.maxconsecutivesuccess = self.consecutivesuccess
        
            
        if (sigma == 0):
            if (mu == self.mean_train):
                self.debugstring = '      BadSigma World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}         scores{}'.format(
                    round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
                    round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
                print(self.debugstring)                
                return 0;

            else:
                sigma = self.stdev_train


            #        pdb.set_trace()
        if(mu < self.mean_train):   #no point computing if world differences are smaller, it may be "much" smaller but that is okay
            self.KL_val = 0   
        else: 
            self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)


        #KLscale = (self.num_epochs + 1 - self.episode / 2) / self.num_epochs  # decrease scale (increase sensitvity)  from start 1 down to  1/2
        #        KLscale = min(1, 4*(1 + self.episode) / num_epochs)  # decrease scale (increase sensitvity)  from start 1 down to  1/2
        KLscale = 1
        dprob = min(1.0, ((KLscale * self.KL_val) *  2**(self.consecutivedynamic-1)))  ## if we have only one dynic failure,  this will scale it by .25  but it doubles each time we get another dynamic failrue in a row
        perfprob = min(1.0, self.PerfScale * PerfKL)  #make this smaller since it is slowly varying and  added every time.. less sensitive (good for FP avoid, sloer l

        #if we had  collisons and not consecuretive valiures, we don't use this episode for dynamic probability .. collisions are not well predicted
        tlen = min(self.scoreforKL,len(self.uccscart.char))
                   


        #random collisions can occur and they destroy probability computation so ignore them
        if( self.character.count("CP") > 4)  and ( "attack" not in self.character) and (self.consecutivefail < 2):
            prob = 0
            print("Debug found ", self.character.count("CP"), "CPs in string")
        else:
            print("Debug did not find CP ", self.character.count("CP"), " in string")            
            prob = min(1,(dprob+perfprob)/2) # use average of dynamic and long-term performance probabilities.


                # infrequent checkto outputting cnts for setinng up wbls for actual cnts to use to see if we whould update world change
        if((self.episode+1) % 10 == 0):
            cntval= np.zeros(15)
            cntprob= np.zeros(15)            
            i=0
            scale= 1./int((self.episode+10) )
            cntval[i] = initcnt = scale*self.trialchar.count("init")
            i+= 1; cntval[i] = blockcnt = scale*self.trialchar.count("Block")
            i+= 1; cntval[i] =blockvelcnt = scale*self.trialchar.count("Block Vel")                
            i+= 1; cntval[i] =polecnt = scale*self.trialchar.count("Pole")
            i+= 1; cntval[i] =cartcnt = scale*self.trialchar.count("Cart")
            i+= 1; cntval[i] =smallcnt = scale*self.trialchar.count("Cart")
            i+= 1; cntval[i] =largecnt = scale*self.trialchar.count("Cart")                                
            i+= 1; cntval[i] =diffcnt = scale*self.trialchar.count("diff")
            i+= 1; cntval[i] =velcnt = scale*self.trialchar.count("Vel")                                
            i+= 1; cntval[i] =failcnt = scale*self.trialchar.count("High")
            i+= 1; cntval[i] =attcart = scale*self.trialchar.count("attacking cart")
            i+= 1; cntval[i] =aimblock = scale*self.trialchar.count("aiming")
            i+= 1; cntval[i] =parallelblock= scale*self.trialchar.count("parallel")
            i+= 1; cntval[i] =SAcnt= scale*self.trialchar.count("SA")
            i+= 1; cntval[i] =HAcnt= scale*self.trialchar.count("HA")                                                                                        


            cntwbl = np.array([ [ 1.44423424, -0.19000001,  0.14582359],
                                [ 0.77655005, -0.13000001,  0.07412372],
                                [ 1.4183835 , -0.08000001,  0.06339877],
                                [ 1.3369761 , -0.06000001,  0.04456476],
                                [ 0.84213063, -0.06000001,  0.03392788],
                                [ 0.84213063, -0.06000001,  0.03392788],
                                [ 1.60734717, -0.10000001,  0.07514313],
                                [ 1.37548511, -0.16000001,  0.11283754],
                                [ 1.18602747, -0.11000001,  0.0627653 ],
                                [ 1.70217044, -0.06000001,  0.05617951],
                                [ 1.50681693, -0.06000001,  0.05125134],
                                [ 1.50681693, -0.06000001,  0.05125134],
                                [ 0.60246664, -0.08000001,  0.0598179 ]
                              ] )

            cntmax=0
            for i in range(12):
                cntprob[i] = self.wcdf(-cntval[i],cntwbl[i,1],cntwbl[i,0],cntwbl[i,2])
                cntmax = max (cntmax,cntprob[i])


            if(cntmax > .1): cntmax=.1;  #limit impact this this is really a cumulative test on things we have already seen
            if(prob < cntmax):
                self.character += 'Using detect as prob. detectcnts: {} dectprob {} '.format(cntval,cntprob)
                prob = max(prob,cntmax)
            else: self.character += 'detectcnts: {} dectprob {} '.format(cntval,cntprob)
                
            
            
            
        if (len(self.problist) < self.scoreforKL):
            self.worldchanged = prob * len(self.problist)/(self.scoreforKL)
        else:
            self.worldchanged = prob
            

#####!!!!!##### end GLue CODE for EVT

#####!!!!!#####  Domain Independent code tor consecurtiv efailures
        failinc = 0
        #if we are beyond KL window all we do is watch for failures to decide if we world is changed
        if(self.episode > self.scoreforKL+1):
            faildiff = self.failcnt/(self.episode+1)-self.failfrac            
            if(faildiff > 0):
                self.character +=  "&" +  "High FailFrac=" + str(self.failcnt/(self.episode+1))
                failinc = max(0,  ((faildiff)*self.failscale)) 
                failinc *= min(1,(self.episode - self.scoreforKL)/self.scoreforKL)   #Ramp it up slowly as its more unstable when it first starts at scoreforKL
                failinc = min(1,failinc)

        #world change blend  can go up or down depending on how probablites vary.. goes does allows us to ignore spikes from uncommon events. as the bump i tup but eventually go down. 
        if(prob < .1 and  self.worldchangedacc > .1 and self.worldchangedacc <.5) : # blend wo  i.e. decrease world change accumulator to limit impact of randome events
            self.worldchangeblend = min(1, (        self.blenddownrate *self.worldchanged + (1-self.blenddownrate) * self.worldchangeblend ))
            self.worldchangedacc = min(1,self.worldchangeblend+failinc)            
            self.debugstring = "BlendDown "                

        else:
            #worldchange acc cannot not go down.. it includes max of old value..
            self.worldchangeblend = min(1, (        self.blenduprate *self.worldchanged + (1-self.blenduprate) * self.worldchangeblend ))

            self.debugstring = "Blendup "                                
            # we add in an impusle each step if the first step had initial world change.. so that accumulates over time

            if(len(self.problist) > 0 ) :
                self.worldchangedacc = min(1,self.problist[0]*(self.initprobscale * 2**(self.consecutiveinit-2)) + max(self.worldchangedacc,self.worldchangeblend+failinc))
            else:
                self.worldchangedacc = min(1,max(self.worldchangedacc,self.worldchangeblend+failinc))
        self.debugstring += '    mu={}, sig {}, mean {} stdev{} vals {} {} thresh {} '.format(
            round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3), round(self.KL_val,3),round(PerfKL,3),  "\n")
            #        print(self.debugstring)
        self.character +=  " " +    self.debugstring
#####!!!!!#####  End Domain Independent code tor consecurtiv efailures


#####!!!!!#####  Start API code tor reporting
        self.character += 'World Change Acc={} {} {}, D/KL Probs={},{}'.format(round(self.worldchangedacc,3), round(self.worldchangeblend,3),round(failinc,3), round(dprob,3), round(perfprob,3))
        self.trialchar += self.character




        if(previous_wc < .5 and self.worldchangedacc        >= .5):
            self.character += "#!#!#!  World change Detected #!#!#!  "

        # if world changed an dour performance is below .8  we start using avoidance reaction
        if(self.worldchangedacc        >= .6 and (100*self.perf/self.totalcnt) < 60) :
                self.uccscart.use_avoid_reaction=True            


            

        if(self.episode == 199):
            initcnt = self.trialchar.count("init")
            blockcnt = self.trialchar.count("Block")
            blockvelcnt = self.trialchar.count("Block Vel")                
            polecnt = self.trialchar.count("Pole")
            cartcnt = self.trialchar.count("Cart")
            smallcnt = self.trialchar.count("Cart")
            largecnt = self.trialchar.count("Cart")                                
            diffcnt = self.trialchar.count("diff")
            velcnt = self.trialchar.count("Vel")                                
            failcnt = self.trialchar.count("High")
            attcart = self.trialchar.count("attacking cart")
            aimblock = self.trialchar.count("aiming")
            parallelblock = self.trialchar.count("parallel")                                                                

            if(self.worldchangedacc        <.5):
                self.character += "##### @@@@@ Ending Characterization of potential observed novelities, but did not declare world novel  with  world change prob: " +str(self.worldchangedacc)
            if(self.worldchangedacc        >= .5):                
                self.character += "##### Ending Characterization of observed novelities in novel world   with  world change prob: " +str(self.worldchangedacc)
            if(initcnt > diffcnt ):
                self.character += " Inital world off and "
            if(diffcnt > initcnt ):
                self.character += " Dynamics of world off and "                    
            if(blockcnt > polecnt and blockcnt > cartcnt ):                    
                self.character += " Dominated by Blocks with"
            if(cartcnt > polecnt and cartcnt >  blockcnt   ):                    
                self.character += " Dominated by Cart with"
            if(polecnt > cartcnt and polecnt > blockcnt ):                    
                self.character += " Dominated by Pole with"
            self.character += " Velocity Violations " + str(velcnt)                                                                                                
            self.character += "; Agent Velocity Violations " + str(blockvelcnt)                
            self.character += "; Cart Total Violations " + str(cartcnt)
            self.character += "; Pole Total Violations " + str(polecnt)
            self.character += "; Speed/position too small Violations " + str(smallcnt)
            self.character += "; Speed/position too  large Violations " + str(largecnt)                                                                                                            
            self.character += "; Attacking Cart Violations " + str(attcart)
            self.character += "; Blocks aiming at blocks " + str(aimblock)                                                                                                            
            self.character += "; Coordinated block motion " + str(parallelblock)
            self.character += "; Agent Total Violations " + str(blockcnt + parallelblock + attcart + blockvelcnt)
            self.character += ";  Violations means that aspect of model had high accumulated EVT model probability of exceeding normal training  "
            if(failcnt > 10):
                self.character += " Uncontrollable dynamics for unknown reasons .. Failure Frequencey too high compared to training"
            self.character += "#####"


                

        return self.worldchangedacc

    def process_instance(self, actual_state):
        #        pertub = (self.cnt > 100) and (self.maxprob < .5)
        pertub = False
        if(self.saveframes):        
            image= actual_state['image']
        self.character += self.uccscart.char  #copy overy any information about collisions
#        if(len(self.uccscart.char)>0): print("Process inst with char", self.uccscart.char)
        probability = self.uccscart.wcprob  # if cart control detected something we start from that estiamte 


#        if("CP in        self.uccscart.char):
#            self.uccscart.lastscore = 0.001111; #  if we had a lot fo collision potential, ignore the score. 
        self.scorelist.append(self.uccscart.lastscore)        
        self.uccscart.char = ""  #reset any information about collisions
        action, expected_state = self.takeOneStep(actual_state,
                                                  self.uccscart,
                                                  pertub)

        # we can now fill in previous history's actual 
        if(self.uccscart.force_action >=0 and self.uccscart.force_action < 5):
            self.uccscart.action_history[self.uccscart.force_action][1] = self.uccscart.format_data(actual_state)        

        #we don't reset in first few steps because random start may be a bad position yielding large score
        #might be were we search for better world parmaters if we get time for that
        #TB.. this does not seem to be needed as reset hasppend when searching for best action 
#        if(self.cnt > self.skipfirstNscores and self.uccscart.lastscore > self.scoretoreset):
#            print("At step ", self.cnt, "resettin to actual because of a large score", self.uccscart.lastscore)
#            self.uccscart.reset(actual_state)



        data_val = self.prev_predict
        self.prev_predict = expected_state
        self.prev_state = actual_state        
        self.cnt += 1
        if (self.cnt == 1):  # if first run cannot check dynamics just initial state
            if(self.uccscart.tbdebuglevel>0):            
                self.debugstring = 'Testing initial state for obvious world changes: actual_state={}, next={}, dataval={}, '.format(actual_state,
                                                                                                                                    expected_state,
                                                                                                                                    data_val)
            initprob= self.istate_diff_EVT_prob(actual_state)

            #update max and add if initprob >0 add list (if =0 itnore as these are very onesided tests and don't want to bias scores in list)
            self.maxprob = max(initprob, self.maxprob)
            if(initprob >0):
                self.problist.append(initprob)  # add a very big bump in prob space so KL will see it
                if(self.uccscart.tbdebuglevel>0):
                    print('Init probability checks set prob to 1 with actual_state={}, next={}, dataval={}, problist={}, '.format(actual_state,
                                                                                                                                expected_state,
                                                                                                                                data_val,
                                                                                                                                self.problist))

                # if (self.debug):
                #     self.debugstring = 'Early Instance: actual_state={}, next={}, dataval={}, '.format(actual_state,expected_state,data_val)
                self.prev_action = action
                return action
        else:

            data_val = self.format_data(data_val)
            prob_values = []
            actual_state = self.format_data(actual_state)

            difference_from_expected = data_val - actual_state  # next 4 are the difference between expected and actual state after one step, i.e.
            current = difference_from_expected


            diffprobability = self.prob_scale * self.cstate_diff_EVT_prob(current)
            probability += diffprobability
            #            pdb.set_trace()            
            #if we have high enough probability and failed often enough and have not searched for pertubations, try searching for action permuations
            #            if(((self.episode < 50) and (self.failcnt/(self.episode+1)-1.5* self.failfrac) > -1000) and self.perm_search< 20):
            # if(True):  #idea not working 
            #     self.perm_search +=1 
            #     #force each possible action into our history as we need them to get mapping ..  this effects the get_best_*_action functions which get the actions and story history
            #     if(self.uccscart.force_action < 5):
            #         self.uccscart.force_action +=  1
            #     else:
            #         actprob = self.try_actions_permutations(actual_state,diffprobability) # try various permuation of actions to see if they can explain the current state.  If it finds one return prob 1 and sets action permutation index in UCCScart.
            #         probability += actprob
            #         if(actprob>0): self.character += "Action search with prob " + str(actprob)
            #         pdb.set_trace()

#####!!!!!#####  end GLUE/API code EVT-
#####!!!!!#####  Start domain dependent adaption

            # if we have not had a lot successess in a row (sign index is right)   and declared world changed and  we ahve enough failures then try another index
            if( False and self.maxconsecutivesuccess < 5 and  self.maxconsecutivefail > self.maxconsecutivefailthresh and  self.consecutivefail > 3 ):
                # try the next permuation.. see if we can reduce the fail rate
                self.uccscart.actions_permutation_index += 1
                if(self.uccscart.actions_permutation_index > (len(self.uccscart.actions_plist)-1)):
                    self.uccscart.actions_permutation_index = 0                    
                self.character += "#####? Too many failures.  Guessing actions were mapped/perturbed.. Now using pertubation " 
                self.character.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
                self.character += "if this is the last time you see this message and performance is now good then characterize this as the action permutation in placeof the  uncontrollable characateration provided after world change provided earlier #####?"
                print(self.character)
                self.consecutivefail = 0
            




            probability = min(1,probability)
            self.problist.append(probability)


            self.maxprob = max(probability, self.maxprob)
            # we can also include the score from control algorithm,   we'll have to test to see if it helps..
            #first testing suggests is not great as when block interfer it raises score as we try to fix it but then it seems novel. 
            #                self.maxprob=min(1,self.maxprob +  self.uccscart.lastscore / self.scalelargescores)
            if (self.cnt > 0 and len(self.problist)>0 ):
                self.meanprob = np.mean(self.problist)

            if (self.uccscart.tbdebuglevel>2):
                self.debugstring = 'Instance: cnt={},actual_state={}, next={},  current/diff={},NovelProb={}'.format(
                    self.cnt, actual_state, expected_state, current, probability)
                print("prob/probval", probability, prob_values, "maxprob", self.maxprob, "meanprob", self.meanprob)  
            
        self.prev_action = action

        if(self.saveframes):        
#            image = feature_vector['image']  #done at begining now
            if image is None:
#                self.log.error('No image received. Did you set use_image to True in TA1.config '
#                               'for cartpole?')
                print('No image received. Did you set use_image to True in TA1.config '
                      'for cartpole?')            
                found_error = True
                
            else:
                s = 640.0 / image.shape[1]
                dim = (640, int(image.shape[0] * s))
                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (10, 30)
                fontScale = .5
                # Blue color in BGR
                if(round(self.worldchangedacc,3) < .5):            color = (255, 0, 0)
                else  :            color = (0,0,255)
                thickness = 2
                fname = '/scratch/tboult/PNG/{1}-Frame-{0:04d}.png'.format(self.framecnt,self.saveprefix)            
                wstring = 'E={4:03d}.{0:03d} RP={7:4.3f} WC={2:4.3f} P{1:3.2f} N={6:.1} C={5:.4},S={3:12.6}'.format(self.uccscart.tick,probability,self.worldchangedacc,self.uccscart.lastscore,self.episode,self.character[-4:], str(self.noveltyindicator),100*self.perf/(max(1,self.totalcnt))        )            
                outimage = cv2.putText(resized, wstring, org, font,
                                       fontScale, color, thickness, cv2.LINE_AA)
                cv2.imwrite(fname, outimage)

                self.framecnt += 1
                if ((self.uccscart.tbdebuglevel>-1 )and self.framecnt < 3):
                    self.debugstring += '  Writing '+ fname + 'with overlay'+ wstring
                    print(self.debugstring)

                
        
        return action



                    
    def mark_tried_actions(self,action, paction):
        '''  mark permutaiton indicies where the current action is in the perturbed action slot has been tried and did not work (used) '''
        for index in range(len(self.uccscart.actions_plist)-1):
            if(self.uccscart.actions_plist[index][action] == paction):
                self.uccscart.actions_permutation_tried[self.uccscart.actions_permutation_index] =1                



    def try_actions_permutations(self,actual_state,diffprob):
        ''' try various permuation of actions to see if they best explain the current state.  If it finds one return prob 1 and sets action permutation index in UCCScart.
         this can be called multiple times because the actual state transition can only use 1 action so if we swap say left-right  we might not see if we need to also swap front/back
        If here action_history should be populated with all actions  
        '''

        return 0        
#####!!!!!#####  end domain dependent adaption

#         #this was an attempt at choosing pertubation based on minimizing the overall probbability  of differences in transitions.. but it did not work well enough.. identifies some but too much noise.  
#         # Maybe revisit when predictions are better
#         diffprobability = np.zeros((5,5))
#         statediff = np.zeros((5,5,13))        
#         for action in  range(5):
#             for index in range(5):
#                 statediff[action][index] = self.uccscart.action_history[action][0] - self.uccscart.action_history[index][1] 
#                 diffprobability[action][index] = self.prob_scale * self.cstate_diff_prob(statediff[action][index])

#         pdb.set_trace()                
#         plen = len(self.uccscart.actions_plist)
#         probs = np.zeros(plen)
#         for index in range(plen):
#             probs[index] = (diffprobability[0][self.uccscart.actions_plist[index][0]]
#                             +diffprobability[1][self.uccscart.actions_plist[index][1]]
#                             +diffprobability[2][self.uccscart.actions_plist[index][2]]
#                             +diffprobability[3][self.uccscart.actions_plist[index][3]]
#                             +diffprobability[4][self.uccscart.actions_plist[index][4]])
            

#         index = np.argmin(probs)
#         minprob = np.min(probs)
#         pdb.set_trace()
#         #if there is a pertubation with lower overall error than        
#         if(minprob < probs[self.uccscart.actions_permutation_index]):
#             self.uccscart.actions_permutation_index = index   #keep that index
#             self.character += "Actions were perturbed.. Now using pertubation " 
#             self.character.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
#             print("Good pertubation ", self.uccscart.actions_plist[self.uccscart.actions_permutation_index],"with index", self.uccscart.actions_permutation_index )
#             self.perm_search = 100   #mark it found
#             return 1
#         else: 
#             print("No pertubation needed was better  than current perm", self.uccscart.actions_plist[self.uccscart.actions_permutation_index],
#                   "with index", self.uccscart.actions_permutation_index )                       
#             return 0
            



#first  try at action remapping 
#         action = self.prev_action
#         # Convert from string to int
#         if action == 'nothing':
#             action = 0
#         elif action == 'right':
#             action = 2    #tb flipped for testing
#         elif action == 'left':
#             action = 1
#         elif action == 'forward':
#             action = 4
#         elif action == 'backward':
#             action = 3



        
#         bestprob = diffprob
#         bestindex = 
# #        self.mark_tried_actions(action,self.uccscart.actions_plist[self.uccscart.actions_permutation_index][action])
# #        pdb.set_trace()        
#         #Run through all permutaitons and mark off those that are inconsitent with current step.
#         #        plen = len(self.uccscart.actions_plist)-1
#         plen = 119
#         while( self.uccscart.actions_permutation_index < plen):
#             self.uccscart.actions_permutation_index += 1
#             if(self.uccscart.actions_permutation_index > len(self.uccscart.actions_plist)):
#                 self.uccscart.actions_permutation_index=0
#             if(self.uccscart.actions_permutation_tried[self.uccscart.actions_permutation_index] >0): continue
#             print("indexs",  self.uccscart.actions_permutation_index, "actions", action,  self.uccscart.actions_plist[self.uccscart.actions_permutation_index][action],
#                   "plist",  self.uccscart.actions_plist[self.uccscart.actions_permutation_index])


#             # If here it has potential, mark it as being tried
#             self.uccscart.actions_permutation_tried[self.uccscart.actions_permutation_index] =1
#             score,state = self.uccscart.one_step_env(self.prev_state,[action])  # this uses current index to see what state results.
#             statediff = self.format_data(state) - actual_state
#             diffprobability = self.prob_scale * self.cstate_diff_prob(statediff)
#             if(diffprobability < bestprob):            
#                 if(diffprobability < .0005): # stop early if good score
#                     self.character += "Actions were perturbed.. Now using pertubation " 
#                     self.character.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
#                     print("Good pertubation ", self.uccscart.actions_plist[self.uccscart.actions_permutation_index],"with index/tried", self.uccscart.actions_permutation_index, self.uccscart.actions_permutation_tried )                    
#                     return 1;
#                 else:
#                     bestprob = diffprobability
#                     bestindex = self.uccscart.actions_permutation_index
           

#         self.uccscart.reset(actual_state)                    #back to normal state
# #        pdb.set_trace()                    
#         if(bestprob  < diffprobability and bestprob < .01): # not great but  better than where we started and maybe good enough. 
#             self.uccscart.actions_permutation_index = bestindex
#             print("Now using pertubation ", self.uccscart.actions_plist[self.uccscart.actions_permutation_index],"with index/tried", self.uccscart.actions_permutation_index, self.uccscart.actions_permutation_tried )
#             self.character += "Actions might be  perturbed.. Now using pertubation "
#             self.character.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
#             self.character += "with prob" + str(bestprob)
#             return bestprob;

#         #else nothing work so reset back to initial action list
#         self.uccscart.actions_permutation_index = 0
#         self.uccscart.actions_permutation_tried = np.zeros(len(self.uccscart.actions_permutation_tried))
        
#         return 0



