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
# import cv2
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
        self.scalelargescores=20
        # takes a while for some randome starts to stabilise so don't reset too early as it
        # reduces world change sensitvity.  Effective min is 1 as need at least a prior state to get prediction.
        self.skipfirstNscores=1

        self.consecutivefail=0
        self.failscale=1.0 #   How we scale failures
        self.failfrac=.30  #Max fail fraction,  when above  this we start giving world-change probability for  failures

        self.initprobscale=.05 #   we scale prob from initial state by this amount (scaled by 2**(consecuriteinit-2) and add world accumulator each time. No impacted by blend this balances risk from going of on non-novel worlds
        self.consecutiveinit=0   # if get consecutitve init failures we keep increasing scale
        self.consecutivedynamic=0   # if get consecutitve dynamic failures we keep increasing scale        

        # Large "control scores" often mean things are off, since we never know the exact model we reset when scores get
        # too large in hopes of  better ccotrol
        self.scoretoreset=200

        #smoothed performance plot for dtection.. see perfscore.py for compuation.  Major changes in control mean these need updated
        self.perflist = []
        self.mean_perf = 0.8883502538071065
        self.stdev_perf = 0.11824239133691708
        self.PerfScale = 0.1    #How much do we weight Performacne KL prob.  make this small since it is slowly varying and added every episode. Small is  less sensitive (good for FP avoid, but yields slower detection). 
        

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
        self.worldchanged = 0
        self.worldchangedacc = 0
        self.blendrate = .3
        
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

        self.imax = self.imin = self.ishape = self.iscale =  None
        self.dmax = self.dshape = self.dscale =  None        

        
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
        
        if(episode ==0):  #reset things that we carry over between episodes withing the same trial
            self.worldchangedacc = 0
            self.failcnt = 0                    
            self.worldchangeblend = 0            
            self.consecutivefail=0
            self.perm_search=0


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

    
    def wcdf(self,x,iloc,ishape,iscale):
        prob = 1-math.pow(math.exp(-abs(x-iloc)/iscale),ishape)
#        if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
        return prob


    def format_data(self, feature_vector):
        # Format data for use with evm
        state = []
        for i in feature_vector.keys():
            if i != 'blocks' and i != 'time_stamp' and i != 'image' and i != 'ticks':
#                print(i,feature_vector[i])
                for j in feature_vector[i]:
                    state.append(feature_vector[i][j])
                #print(state)
        return np.asarray(state)

    # get probability differene froom initial state
    def istate_diff_prob(self,actual_state):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart x'" , " Cart y'" , " Cart z'" ,  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole x'" , " Pole y'" , " Pole z'" , " Block x" , " Block y" , " Block x" ,  " Block x'" , " Block y'" , " Block z'" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]
        
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
                                  2.8e-02, 2.8e-02, 5.800000e-04,  # pole vel
                                  9.50+00, 9.50+00, 9.50,  # block pos
                                  10.0, 10.0, 10.0, #block vel
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
                                  -4.090533e+00, -4.090533e+00, -4.0090533,  # block pos
                                  5.1, 5.1, 5.1, #block vel  not quite as programmed as it can drop with gravity on first step which we we don't see
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
                                 [7.19656366e-01, 0.00000000e+00, 1.14079390e-02],
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

            # self.imax = np.array([ 2.9992560e+00,  2.9979550e+00,  0.0000000e+00,  2.0129000e-02,
            #                     1.9909000e-02,  0.0000000e+00,  9.9970000e-03,  9.9890000e-03,
            #                     9.9930000e-03,  1.0000000e+00,  2.9892000e-02,  2.9460000e-02,
            #                     6.9400000e-04,  4.1599750e+00,  4.1772000e+00,  9.1959550e+00,
            #                     1.4164505e+01,  1.3935462e+01,  1.3447222e+01, 5.0000000e+00,
            #                     5.0000000e+00,  0.0000000e+00,  5.0000000e+00, 5.0000000e+00,
            #                     0.0000000e+00,  5.0000000e+00,  5.0000000e+00,  0.0000000e+00,
            #                     5.0000000e+00,  5.0000000e+00,  0.0000000e+00, 5.0000000e+00,
            #                     5.0000000e+00,  1.0000000e+01,  5.0000000e+00, 5.0000000e+00,
            #                     1.0000000e+01,  5.0000000e+00,  5.0000000e+00,  1.0000000e+01,
            #                     5.0000000e+00,  5.0000000e+00,  1.0000000e+01])

        
            # initwbl = np.array([[[3.78224871e-01, 0.00000000e+00, 6.42222350e-03],
            #                      [1.45674772e+00, 0.00000000e+00, 7.22685918e-03],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.97583085e-01, 0.00000000e+00, 1.75580159e-04],
            #                      [1.96821265e-01, 0.00000000e+00, 9.70341399e-05],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [2.02800383e+00, 0.00000000e+00, 3.99891262e-05],
            #                      [3.03801773e+00, 0.00000000e+00, 4.40613359e-05],
            #                      [1.51058401e+00, 0.00000000e+00, 2.94744087e-05],
            #                      [1.00000000e+00, 0.00000000e+00, 8.31479419e-07],
            #                      [1.82762278e+00, 0.00000000e+00, 1.78638160e-03],
            #                      [3.15406134e-01, 0.00000000e+00, 1.36080652e-03],
            #                      [1.70643986e+00, 0.00000000e+00, 6.31591494e-05],
            #                      [7.64714520e-01, 0.00000000e+00, 2.64113853e-02],
            #                      [1.18910528e+00, 0.00000000e+00, 3.46403602e-02],
            #                      [1.18949256e+00, 0.00000000e+00, 3.22099289e-02],
            #                      [4.43258952e-01, 0.00000000e+00, 2.21087674e-01],
            #                      [3.77528575e-01, 0.00000000e+00, 1.25617083e-01],
            #                      [3.79085005e-01, 0.00000000e+00, 5.70591679e-02],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
            #                     [[1.14647315e+03, 0.00000000e+00, 5.98184712e+00],
            #                      [7.95907540e+02, 0.00000000e+00, 5.97032366e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [3.85856082e+02, 0.00000000e+00, 3.98647980e-02],
            #                      [5.27281416e+02, 0.00000000e+00, 3.97249433e-02],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [8.37233758e+02, 0.00000000e+00, 1.99057524e-02],
            #                      [1.11290674e+03, 0.00000000e+00, 1.99002680e-02],
            #                      [1.71697736e+03, 0.00000000e+00, 1.99088640e-02],
            #                      [2.48200138e+01, 0.00000000e+00, 1.31926675e-04],
            #                      [6.57437394e+01, 0.00000000e+00, 5.63167042e-02],
            #                      [3.99549848e+01, 0.00000000e+00, 5.58715873e-02],
            #                      [2.38534737e+01, 0.00000000e+00, 1.25244810e-03],
            #                      [2.67750496e+02, 0.00000000e+00, 8.22415581e+00],
            #                      [6.77758757e+02, 0.00000000e+00, 8.25039397e+00],
            #                      [3.98975942e+02, 0.00000000e+00, 8.27973861e+00],
            #                      [1.16961102e+01, 0.00000000e+00, 2.13144107e+01],
            #                      [6.43380302e+00, 0.00000000e+00, 2.21857795e+01],
            #                      [1.74231073e+03, 0.00000000e+00, 1.97013752e+01],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

            self.ishape = initwbl[0,:,0]
            self.iscale = initwbl[0,:,2]

        imax = self.imax            
        imin = self.imin
        ishape = self.ishape
        iscale = self.iscale        
            

        
        initprob=0  # assume nothing new in world
        #check if any abs (block velocities) < 6 > 10 in any dimension   Done as a hack.. better to do EVT fitting on  values based on training ranges. 
        cart_pos = [actual_state['cart']['x_position'],actual_state['cart']['y_position'],actual_state['cart']['z_position']]
        cart_pos = np.asarray(cart_pos)

        charactermin=1e-3
        
        istate = self.format_istate_data(actual_state)
        # do base state for cart(6)  and pole (7) 
        for j in range (13):
            if(abs(istate[j]) > imax[j]):
                probv =  self.wcdf(istate[j],imax[j],iscale[j],ishape[j]);
#                probv=  (istate[j] - imax[j]) / (abs(istate[j]) + abs(imax[j]))                
                if(probv>charactermin and len(self.character) < 256):
                    initprob += probv
                    self.character +=  str(dimname[j]) + " init too large  " + str(round(istate[j],3)) +" " + str(round(imax[j],3)) +" " + str(round(probv,3))

                
            if(abs(istate[j]) < imin[j]):
                probv =  self.wcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);
                if(probv>charactermin and len(self.character) < 256):
                    initprob += probv
                    self.character += str(dimname[j]) + " init too small  " + str(round(istate[j],3)) +" " + str(round(imin[j],3)) +" " + str(round(probv,3))

        
        wallstart= len(istate) - 24 
#         k=19 # for name max/ame indixing where we have only one block
#         for j in range (wallstart,len(istate),1):
#             if(j >= imax.shape[0] or j >= iscale.shape[0] ): break
#             if(istate[j] > imax[k]):
#                 probv =  self.wcdf(abs(istate[j]),imax[j],iscale[j],ishape[j]);
# #                probv=  (istate[j] - imax[k])/ (abs(istate[j]) + abs(imax[k]))
#                 if(probv>charactermin and len(self.character) < 256):
#                     initprob += probv
#                     self.character += str(dimname[k])+ " init wall too large  " + " " + str(round(istate[j],3)) +" " + str(round(imax[k],3))+" " + str(round(probv,3))
#             if(istate[j] < imin[k]):
#                 probv =  self.wcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);
# #                probv=  (imin[k] - istate[j])/ (abs(istate[j]) + abs(imin[k]))
#                 if(probv>charactermin and len(self.character) < 256):
#                     initprob += probv
#                 self.character += str(dimname[k]) + " init wall too small  " + " " + str(round(istate[j],3)) +" " + str(round(imin[k],3))+" " + str(round(probv,3))
#             k = k +1

                    
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,wallstart,1):
            if(abs(istate[j]) > imax[k]):
                probv =  self.wcdf(abs(istate[j]),imax[j],iscale[j],ishape[j]);
#               probv =  (istate[j] - imax[k]) / (abs(istate[j]) + abs(imax[k]))                
                if(probv>charactermin and len(self.character) < 256):
                    initprob += probv
                    self.character += str(dimname[k])+ " init block too large " +  " " + str(round(istate[j],3)) +" " + str(round(imax[k],3)) +" " + str(round(probv,3))
            if(abs(istate[j]) < imin[k]):
                probv =  self.wcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);
#                probv=  (imin[k] - istate[j])/ (abs(istate[j]) + abs(imin[k]))
                if(probv>charactermin and len(self.character) < 256):
                    initprob += probv
                    self.character += " " + str(dimname[k]) + " init block too small " +  " " + str(round(istate[j],3)) +" " + str(round(imin[k],3)) +" " + str(round(probv,3))
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
    def cstate_diff_prob(self,cdiff):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart x'" , " Cart y'" , " Cart z'" ,  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole x'" , " Pole y'" , " Pole z'" , " Block x" , " Block y" , " Block x" ,  " Block x'" , " Block y'" , " Block x'" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]

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

            # np.array([ 0.056383,  0.056362,  0.      ,  1.391068,  1.731418,  0.      ,
            #                           0.069045,  0.1484  ,  0.155603,  1.11127 , 15.884814, 13.092445,
            #                           17.902195,  0.170244,  0.18108 ,  0.213811,  8.512197,  9.053995,
            #                           10.690558])

            # diffwbl = np.array([[[4.31394240e-01, 0.00000000e+00, 8.00241370e-03],
            #                      [3.92759789e+00, 0.00000000e+00, 1.00268135e-02],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [3.29478359e-01, 0.00000000e+00, 6.91484083e-02],
            #                      [3.35595225e-01, 0.00000000e+00, 2.39486210e-02],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [1.05788918e+00, 0.00000000e+00, 1.27401283e-02],
            #                      [5.74234617e-01, 0.00000000e+00, 1.47940161e-02],
            #                      [1.43991512e+00, 0.00000000e+00, 4.05398364e-02],
            #                      [6.55524978e-01, 0.00000000e+00, 1.24303115e-01],
            #                      [1.02330558e+00, 0.00000000e+00, 3.02142403e+00],
            #                      [9.77731333e-01, 0.00000000e+00, 2.33579452e+00],
            #                      [1.60333044e+00, 0.00000000e+00, 1.84570151e+00],
            #                      [3.21389727e+00, 0.00000000e+00, 1.88286180e-02],
            #                      [1.34330615e+00, 0.00000000e+00, 2.54176133e-02],
            #                      [1.30368325e+00, 0.00000000e+00, 3.59191451e-02],
            #                      [3.21398155e+00, 0.00000000e+00, 9.41423729e-01],
            #                      [1.34327427e+00, 0.00000000e+00, 1.27085628e+00],
            #                      [1.30368758e+00, 0.00000000e+00, 1.79586015e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
            #                     [[1.59922572e+01, 0.00000000e+00, 6.08061393e-02],
            #                      [2.07902979e+01, 0.00000000e+00, 8.62744624e-02],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [6.27580542e-01, 0.00000000e+00, 2.24573798e-02],
            #                      [5.24863381e-01, 0.00000000e+00, 6.64975959e-02],
            #                      [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            #                      [3.11972791e+00, 0.00000000e+00, 7.42374787e-02],
            #                      [9.11047920e+00, 0.00000000e+00, 2.99352992e-02],
            #                      [9.81315330e-01, 0.00000000e+00, 4.05332430e-01],
            #                      [2.85966393e+01, 0.00000000e+00, 1.55589272e-01],
            #                      [1.41975186e+01, 0.00000000e+00, 3.28902023e+00],
            #                      [2.15465624e+01, 0.00000000e+00, 3.92385061e+00],
            #                      [1.34901067e+00, 0.00000000e+00, 5.33799457e+01],
            #                      [1.37353995e+01, 0.00000000e+00, 3.22496309e-01],
            #                      [2.31717538e+01, 0.00000000e+00, 3.04926977e-01],
            #                      [2.27726846e+01, 0.00000000e+00, 2.72298682e-01],
            #                      [1.37351558e+01, 0.00000000e+00, 1.61248042e+01],
            #                      [2.31718613e+01, 0.00000000e+00, 1.52463323e+01],
            #                      [2.27720116e+01, 0.00000000e+00, 1.36149254e+01],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            #                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])
            self.dshape = diffwbl[0,:,0]
            self.dscale = diffwbl[0,:,2]        
        imax = self.dmax;
        imin = -imax
        ishape = self.dshape
        iscale = self.dscale

        #else  just initize locals

#        if(cdiff[0] > .01):
#            pdb.set_trace()
        
        #check if any abs (block velocities) < 6 > 10 in any dimension   Done as a hack.. better to do EVT fitting on  values based on training ranges. 
        if(self.cnt < self.skipfirstNscores): return 0;    #need to be far enough along to get good prediciton

        prob=0  #where we accumualte probability
        charactermin=1e-3
        istate = cdiff
        # do base state for cart(6)  and pole (7) 
        for j in range (13):        
            if(istate[j] > imax[j]):
#                probv =  self.wcdf(istate[j],imax[j],iscale[j],ishape[j]);
                probv =  self.wcdf(abs(istate[j]),imax[j],iscale[j],ishape[j])                
                if(probv>charactermin and len(self.character) < 256):
                    self.character += dimname[j] + " diff too large prob " + " " + str(round(probv,5)) + " s/l " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
                prob += probv
            elif(istate[j] < imin[j]):
                probv =  self.wcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);                
                if(probv>charactermin and len(self.character) < 256):
                    self.character += dimname[j] + " diff too small prob " + " " + str(round(probv,5)) + "  s/l " + str(round(istate[j],5)) +  " " + str(round(imin[j],5))
                prob += probv
        

        #no walls in dtate diff just looping over blocks
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,len(istate),1):
            if(istate[j] > imax[k]):
                probv =  self.wcdf(abs(istate[j]),imax[j],iscale[j],ishape[j]);                
                if(probv>charactermin and len(self.character) < 256):
                    self.character += " " + str(dimname[k]) + " diff too large, prob " + " "
                    + str(round(probv,5)) + "  s/l " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
                    prob += probv
            elif(istate[j] < imin[k]):
                probv =  self.wcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);                
                if(probv>charactermin and len(self.character) < 256):
                    self.character += " " + str(dimname[k]) + " diff too small, prob " + " "
                    + str(round(probv,5)) + " s/l " + str(round(istate[j],5)) +  " " + str(round(imin[j],5))
                prob += probv
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
    
                    

    def try_actions_permutations(self,actual_state,diffprob):
        ## try various permuation of actions to see if they can explain the current state.  If it finds one return prob 1 and sets action permutation index in UCCScart.
        ## this can be called multiple times because the actual state transition can only use 1 action so if we swap say left-right  we might not see if we need to also swap front/back
        stepstaken = [self.prev_action]
        bestprob = diffprob
        bestindex = self.uccscart.actions_permutation_index
        for index in range(len(self.uccscart.actions_plist)-1):
            self.uccscart.actions_permutation_index += 1
            score,state = self.uccscart.one_step_env(self.prev_state,stepstaken)
            current = self.format_data(state) - actual_state
            diffprobability = self.prob_scale * self.cstate_diff_prob(current)
            if(diffprobability < bestprob):            
                if(diffprobability < .0005): # stop early if good score
                    self.character += "Actions were perturbed.. Now using pertubation "
                    self.character.join(self.uccscart.actions_plist[self.uccscart.actions_permutation_index])                    
                    return 1;
                else:
                    bestprob = diffprobability
                    bestindex = index
           
                    
        if(bestprob  < diffprobability and bestprob < .01): # not great but  better than where we started and maybe good enough. 
            self.uccscart.actions_permutation_index = bestindex            
            self.character += "Actions might be  perturbed.. Now using pertubation "
            self.character.join(self.uccscart.actions_plist[self.uccscart.actions_permutation_index])
            self.character += "with prob" + str(bestprob)
            return bestprob;

        #else reset back to initial action list
        self.uccscart.actions_permutation_index = 0 
    


    def world_change_prob(self,settrain=False):
        mlength = len(self.problist)
        mlength = min(self.scoreforKL,mlength)
        # we look at the larger of the begging or end of list.. world changes most obvious at the ends. 

        window_width=11
        if (len(self.perflist) >2*self.scoreforKL):   #look at list of performacne to see if its deviation from training is so that is.. skip more since it needs to be stable for window smoothing+ mean/variance computaiton
            #get smoothed performance 
            cumsum_vec = np.cumsum(np.insert(self.perflist, 0, 0))
            smoothed = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
            pmu = np.mean(smoothed[:-self.scoreforKL])
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
            mu = np.mean(self.problist[1:mlength-1])
            sigma = np.std(self.problist[1:mlength-1])
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
           

        #        if (len(self.problist) < 3):
        #            print("Very short, world must have changed")
        #            return 1;
        if (len(self.problist) < 198):   #for real work
            self.failcnt += 1
            if( self.consecutivefail >0): self.consecutivefail += 1
            else: self.consecutivefail=1

            self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)                
            self.debugstring = '   ***Short World Change Acc={}, Failcnt= {} Prob ={},,mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,3),self.failcnt, [round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
                round(self.stdev_train,3) ,round(self.KL_val,5), round(self.KL_threshold,5), "\n", [round(num,2) for num in self.scorelist])
            if (self.debug):
                print(self.debugstring)
        else:
            self.consecutivefail=0
            
        if (sigma == 0):
            if (mu == self.mean_train):
                self.debugstring = '      World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}         scores{}'.format(
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
        dprob = min(1.0, KLscale * self.KL_val) *  2**(self.consecutivedynamic-2)  ## if we have only one dynic failure,  this will scale it by .125  but it doubles each time we get another dynamic failrue in a row
        perfprob = min(1.0, self.PerfScale * PerfKL)  #make this smaller since it is slowly varying and  added every time.. less sensitive (good for FP avoid, sloer l

        #if we had  collisons and not consecuretive valiures, we don't use this episode for dynamic probability .. collisions are not well predicted
        if( "CPCP" in  self.uccscart.char and self.consecutivefail < 2):
            prob = 0
        else:
            prob = min(1,(dprob+perfprob)/2) # use average of dynamic and long-term performance probabilities.

            
        if (len(self.problist) < self.scoreforKL):
            self.worldchanged = prob * len(self.problist)/(self.scoreforKL)
        else:
            self.worldchanged = prob
            

        #world change blend  can go up or down depending on how probablites vary.. goes does allows us to ignore spikes from uncommon events. as the bump i tup but eventually go down. 
        #worldchange acc cannot not go down.. it includes max of old value..
        self.worldchangeblend = min(1, (        self.blendrate *self.worldchanged + (1-self.blendrate) * self.worldchangeblend ))
        #final result is monotonicly increasing, and we add in an impusle each step if the first step had initial world change.. so that accumulates over time


        failinc = 0
        #if we are beyond KL window all we do is watch for failures to decide if we world is changed
        if(self.episode > self.scoreforKL+1):
            if(self.failcnt/self.episode > self.failfrac):
                self.character += "High FailFrac=" + str(self.failcnt/(self.episode+1))
                failinc = max(0,  (self.failcnt/(self.episode+1)-self.failfrac)*self.failscale )
                failinc *= min(1,(self.episode - self.scoreforKL)/self.scoreforKL)   #May ramp it up slowly as its more unstable when it first starts
            
        if(len(self.problist) > 0 ) :
            self.worldchangedacc = min(1,self.problist[0]*(self.initprobscale * 2**(self.consecutiveinit-2)) + max(self.worldchangedacc,self.worldchangeblend+failinc))
        else:
            self.worldchangedacc = min(1,max(self.worldchangedacc,self.worldchangeblend+failinc))
        self.debugstring = '      World Change Acc={}, KLprobs={},{}, mu={}, sig {}, mean {} stdev{} vals {} {} thresh {}  Problist ={}, {}  scores{}'.format(
            round(self.worldchangedacc,3), round(dprob,3), round(perfprob,3),round(mu,3), round(sigma,3), round(self.mean_train,3),
            round(self.stdev_train,3), round(self.KL_val,3),round(PerfKL,3),  "\n",[round(num,4) for num in self.problist],"\n", [round(num,2) for num in self.scorelist])
        print(self.debugstring)                
        self.character += 'World Change Acc={} {} {}, D/KL Probs={},{}'.format(round(self.worldchangedacc,3), round(self.worldchangeblend,3),round(failinc,3), round(dprob,3), round(perfprob,3))

        return self.worldchangedacc

    def process_instance(self, actual_state):
        #        pertub = (self.cnt > 100) and (self.maxprob < .5)
        pertub = False
        self.character += self.uccscart.char  #copy overy any information about collisions
#        if(len(self.uccscart.char)>0): print("Process inst with char", self.uccscart.char)
        probability = self.uccscart.wcprob  # if cart control detected something we start from that estiamte 


#        if("CP in        self.uccscart.char):
#            self.uccscart.lastscore = 0.001111; #  if we had a lot fo collision potential, ignore the score. 
        self.scorelist.append(self.uccscart.lastscore)        
        self.uccscart.char = ""  #reset any information about collisions
        action, expected_state = self.takeOneStep(actual_state, self.uccscart, pertub)
        

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
            self.debugstring = 'Testing initial state for obvious world changes: actual_state={}, next={}, dataval={}, '.format(actual_state,
                                                                                                                                    expected_state,
                                                                                                                                    data_val)
            initprob= self.istate_diff_prob(actual_state)

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


            diffprobability = self.prob_scale * self.cstate_diff_prob(current)
            probability += diffprobability
            
            #if we have high enough probability and failed often enough and have not searched for pertubations, try searching for action permuations
            if(diffprobability > .6 and self.consecutivefail > 3 and self.perm_search< 5):
                self.perm_search += 1
                actprob = self.try_actions_permutations(actual_state,diffprobability) # try various permuation of actions to see if they can explain the current state.  If it finds one return prob 1 and sets action permutation index in UCCScart. 
                probability += actprob


#            if(probability>0): probability=1
            self.problist.append(probability)
            
#            if(self.given):
#                probability=1
#                self.uccscart.lastscore  = self.uccscart.lastscore *10    #make the scores highrer so we tend to use the more exensive twostep                 


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
        return action
