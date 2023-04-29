# UCCS TA 2 helper
import pdb
import numpy as np
import gym
from gym import make

import os
import sys
import random
import time
#from utils import rollout
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
import cv2
import PIL
import torch
import json
import argparse
from collections import OrderedDict
from functools import partial
from torch import Tensor
import torch.multiprocessing as mp
#from my_lib import *
from statistics import mean
import scipy.stats as st

import gc
import random
import csv
import importlib.util
import math

from datetime import datetime, timedelta



def heatmap(data, width,height):
    data = 255 * data
    data[data>255] = 255
    data = data.astype(np.uint8)
    heat = cv2.resize(data,(width,height))
    heat = cv2.cvtColor(heat,cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat
                                    






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
        self.scoreforKL=19        # we only use the first sets of scores for KL because novels worlds close when balanced .  
        self.num_epochs = 200
        self.num_dims = 4
        self.num_blocks = None
        self.scalelargescores=20
        # takes a while for some randome starts to stabilise so don't reset too early as it
        # reduces world change sensitvity.  Effective min is 1 as need at least a prior state to get prediction.
        self.skipfirstNscores=1
        self.maxinitprob=4  # both max for per episode individual prob as well as prob scale.  
        self.current_state=None
#        self.statelist=np.empty(300, dtype=object)
        self.probvector= np.zeros(500)
        self.nextprob=15

        self.blockmin=999
        self.blockmax=-999
        self.blockvelmax=-999        
        self.normblockmin=999
        self.normblockmax=-999        

        self.maxconsecutivefailthresh= 4  # if we see this many in a row we declare world changed as we never see even 3 in training 
        
        # we penalize for high failure rantes..  as  difference (faildiff*self.failscale) )
        self.failscale=8.0 #   How we scale failure fraction.. can be larger than one since its fractional differences and genaerally < .1 mostly < .05
        self.failfrac=.18  #Max fail fraction,  when above  this we start giving world-change probability for  failures
        self.maxfailfrac=.25  #Max fail fraction,  when above  this we start giving world-change probability for  failures
        
        # because of noisy simulatn and  many many fields and its done each time step, we limit how much this can add per time step
        self.maxdynamicprob = .15  # was .175 but too many false detects on non-novel trials so reduced it a bit. .  Added separate for cart/pole.. balls seem more stable so inreased to .5
        self.maxclampedprob = .005  # because of broken simulator we get randome bad value in car/velocity. when we detect them we limit their impact to this ..
        self.clampedprob =   self.maxclampedprob       
        self.cartprobscale=.25 #   we scale prob from cart/pole because the environmental noise, if we fix it this will make it easire to adapt .
#        self.initprobscale=1.0 #   we scale prob from initial state by this amount (scaled as consecuriteinit increases) and add world accumulator each time. No impacted by blend this balances risk from going of on non-novel worlds
        self.initprobscale=.5 #   Were getting too many detects on no-novel worlds.. so reduce
        self.consecutiveinit=0   # if get consecutitve init failures we keep increasing scale
        self.dynamiccount=0   # if get consecutitve dynamic failures we keep increasing scale
        self.consecutivewc=0   # if get consecutitve world change overall we keep increasing scale
        self.ballprobscale=.5   # How much  do we scale ball location error probability  sum 0.5 would be average, larger increased mixed sensitivity
        self.ballprobscale=1.1   # How much  do we scale max ball location error probability
        self.maxballscaled = 1
        self.levelcnt=np.zeros(10)

        # Large "control scores" often mean things are off, since we never know the exact model we reset when scores get
        # too large in hopes of  better ccotrol
        self.scoretoreset=1000

        #smoothed performance plot for dtection.. see perfscore.py for compuation.  Major changes in control mean these need updated
        self.perflist = []
        self.mean_perf = 0.8883502538071065
        self.stdev_perf = 0.0824239133691708
        self.PerfScale = 0.15    #How much do we weight Performacne KL prob.  make this small since it is slowly varying and added every episode. Small is  less sensitive (good for FP avoid, but yields slower detection). 

        self.consecutivesuccess=0
        self.consecutivefail=0
        self.maxconsecutivefail=0
        self.maxconsecutivesuccess=0
        self.consecutivehighball=0
        self.minprob_consecutive = .1
        self.mindynprob = .01
        self.dynblocksprob = 0                   
        assert(self.minprob_consecutive <= self.maxdynamicprob)  # should be not be larger than maxdynamicprob
        self.tick=0

        self.maxcarlen=25600

        # TODO: change evm data dimensions
        if (self.num_dims == 4):
            self.mean_train = 0
            self.stdev_train = 0.0
            self.dynam_prob_scale = 2  # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
        else:
#            self.mean_train = .198   #these are old values from Phase 1 2D cartpole..  for Pahse 2 3D we compute frm a training run.
#            self.stdev_train = 0.051058052318592555
            self.mean_train = 0.004   
            self.stdev_train = 0.009
        self.dynam_prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.

        self.cnt = 0
        self.framecnt = 0
        self.saveframes = False
        self.saveprefix = random.randint(1,10000)
        
        self.worldchanged = 0
        self.worldchangedacc = 0
        self.previous_wc = 0        
        self.blenduprate = 1           # fraction of new prob we use when blending up..  It adapts over time
        self.blenddownrate = .25        # fraction of new prob we use when blending down..  should be less than beld up rate.  No use of max
        self.minblenddownrate = .1        # fraction of new prob we use when blending down..  should be less than beld up rate.  No use of max        
        
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
        self.debug = False
        self.debug = True        
        self.debugstring = ""
        self.logstr=""
        self.summary=""        
        self.hint=""        
        self.trialchar=""        

        self.imax = self.imin =   self.imean = self.istd = self.ishape = self.iscale =  None
        self.dmax = self.dshape = self.dscale = self.dmean = self.dstd =   None        


        if(False):                         
            #            self.uccscart.reset(actual_state)
            ldist = self.line_to_line_dist(np.array([0,0,1]),np.array([0,0,-1]),np.array([0,1,0]),np.array([0,-1,0]))
            ldist2 = self.line_to_line_dist(np.array([0,0,1]),np.array([0,0,-1]),np.array([0,1,0]),np.array([0,-1,.001]))
            pdist = self.point_to_line_dist(np.array([0,0,1]), np.array([0,0,0]), np.array([0,1,0]))
            pdist2 = self.point_to_line_dist(np.array([0,0,0]), np.array([0,2,0]), np.array([0,1,0]))            
            print("line distances", ldist, ldist2, "Pointdist",pdist,pdist2)
        
        
        # Create prediction environment
        env_location = importlib.util.spec_from_file_location('CartPoleBulletEnv', \
                                                              'cartpolepp/UCart.py')
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
#        self.statelist=np.empty(300, dtype=object)
        self.given = False
        self.maxprob = 0
        self.meanprob = 0
        self.cnt = 0
        self.logstr=""
        self.debugstring=""        
        self.episode = episode
        self.worldchanged = 0
        self.uccscart.resetbase()
        self.uccscart.reset()
        self.uccscart.episode=episode
        self.dynblocksprob = 0                   
        self.dynamiccount=0            
        self.probvector= np.zeros(500)
        self.nextprob=15


        if(episode == 0):
            self.blockmin=999
            self.blockmax=-999
            self.blockvelmax=-999
            self.normblockmin=999
            self.normblockmax=-999        


        if(episode <10):
            self.normblockmin=min(self.blockmin,self.normblockmin)  #collect data about min/max seen by blocks in first episodes
            self.normblockmax=max(self.blockmax,self.normblockmax)
            if(self.uccscart.tbdebuglevel>1): print("PreNovelty Episode"+ str(episode)+ "Minmax block norms" ,self.blockmin,self.blockmax,self.normblockmin,self.normblockmax)
        elif(episode < self.scoreforKL):  #reset things that we carry over between episodes withing the same trial.. but also need at least enough for KL
            self.worldchangedacc = 0
            self.uccscart.wcprob=0            
            self.failcnt = 0                    
            self.worldchangeblend = 0
            self.consecutivefail=0
            self.perm_search=0
            self.trialchar=""
            self.uccscart.characterization={'level': None, 'entity': None, 'attribute': None, 'change': None}                    
            self.clampedprob = self.maxclampedprob/2  # don't use the noisy stuff much at very begining, to avoid FP going early
            self.blenduprate = 1           # fraction of new prob we use when blending up..  It adapts over time
            self.blockmin=999
            self.blockmax=-999
            self.blockvelmax=-999
#        elif(episode < 2*self.scoreforKL):  #long term the noisy/bad probabilities seem to grow so reduce the max they can impact  
        elif(episode < 3*self.scoreforKL):  #Phase 3 is not as noisy.. could use hints to change or a develop a noise model
            self.clampedprob = self.maxclampedprob  *  ((2*self.scoreforKL-episode)/(self.scoreforKL))**2  # we reduce max from noisy ones over the window size
            self.blenduprate = 1           # fraction of new prob we use when blending up..  It adapts over time            
            self.blockmin=999
            self.blockvelmax=-999            
            self.blockmax=-999
        else:
            self.clampedprob = 0
            self.blenduprate = max(.1,(3*self.scoreforKL-episode)/(self.scoreforKL))          # fraction of new prob we use when blending up..  It adapts over time
         #self.blenddownrate = max(self.minblenddownrate,min(.5,.5*(2*self.scoreforKL-episode)/(self.scoreforKL)))
            self.blenddownrate = min(.5,self.blenddownrate + .05)            
        
        if(episode > 3*self.scoreforKL):  #stop blending once we have stable KL values, and don't search since its expensive but cannot be useful after that many
            self.failcnt = 0                    
            self.worldchangeblend = 0
            self.failcnt = 0                                
            self.consecutivefail=0
            self.perm_search=0

        if(self.uccscart.wcprob == 1):
            self.worldchangedacc=1
            self.worldchanged=1            
            self.worldchangeblend=1

        if(self.worldchangedacc >.5):
            self.uccscart.wcprob=max(self.uccscart.wcprob,self.worldchangedacc)                        

            
            



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
        self.tick = self.uccscart.tick

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
#        if(prob > 1e-4): print("in rwcdf",round(prob,6),x,iloc,ishape,iscale)
        return prob

    #abs wbl for unsided fitting    
    def awcdf(self,x,iloc,ishape,iscale):
        prob = 1-math.pow(math.exp(-abs(x-iloc)/iscale),ishape)
#        if(prob > 1e-4): print("in awcdf",round(prob,6),x,iloc,ishape,iscale)
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
            if i == 'time_stamp' or  i == 'image' or i == 'hint': continue             
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
            if i == 'cart' or i == 'pole'  :                
#                print(i,feature_vector[i])
                for j in feature_vector[i]:
                    state.append(feature_vector[i][j])
                #print(state)
            elif i == 'blocks':
                for block in feature_vector[i]:
                    for key in block.keys():
                        if key != 'id':
                            state.append(block[key])
                
        return np.asarray(state)

    def point_to_line_dist(self,cpos, bpos, bvel):
        pdiff = np.subtract(cpos ,bpos)
        nval = np.linalg.norm(bvel)
        if(nval >0) :
            dist =  np.linalg.norm(np.cross(pdiff,bvel))/ nval
        else: dist = np.linalg.norm(pdiff)  # if vector direction (velocity) is 0 then distance is distance between the two points
        return dist

    #line to line give position and direction (velocity) representation -- if velocity is 0 return 9999 as distance (its not well defined  but this will make it look novel)
    def line_to_line_dist(self,apos,avel, bpos, bvel):

        magA = np.linalg.norm(avel)
        magB = np.linalg.norm(bvel)
        if(magA ==0 or magB==0): return 9999;
        
        nA = avel / magA
        nB = bvel / magB
        
        cross = np.cross(nA, nB);
        denom = np.linalg.norm(cross)**2
        t = (bpos-apos)
        d0 = np.dot(nA,t)
        
        if(denom == 0) :  # parallel lines
            dist =  np.linalg.norm(((d0*nA)+apos)-bpos)
        else:
            
            # skew lines: Calculate the projected closest points
            t = (bpos - apos);
            detA = np.linalg.det([t, nB, cross])
            detB = np.linalg.det([t, nA, cross])
            
            t0 = detA/denom;
            t1 = detB/denom;
            
            pA = apos + (nA * t0) # Projected closest point on line A
            pB = bpos + (nB * t1) # Projected closest point on line B
            dist =    np.linalg.norm(pA-pB)

        return dist
    

    def block_pos(self,istate,blocknum):
        return istate[13+blocknum*6:13+blocknum*6+3]

    def block_vel(self,istate,blocknum):
        return istate[13+blocknum*6+3:13+blocknum*6+6]

    def cart_pos(self,istate):
        return istate[0:3]

    def cart_vel(self,istate):
        return istate[3:6]    

    def pole_pos(self,istate):
        return istate[6:9]

    def pole_vel(self,istate):
        return istate[9:12]    

    def unit_vector(self,vector):
        """ Returns the unit vector of the vector  and if norm is too small  limit divsiion to avoid numeric instability  """
        return vector / max(np.linalg.norm(vector),1e-16)

    def vector_angle(self,v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> vector_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            >>> vector_angle((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> vector_angle((1, 0, 0), (1, 0, 0))
            0.0
        """
        v1 = self.unit_vector(v1)
        v2 = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        

    # get probability differene froom initial state
    def istate_diff_EVT_prob(self,actual_state):
        dimname=[" x Cart" , " y Cart" , " z Cart" ,  " x Cart Vel" , " y Cart Vel" , " z Cart Vel ",  " x Pole" , " y Pole" , " z Pole" ," w Pole" ,  " x Pole Vel" , " y Pole Vel" , " z Pole Vel" , " z Block " , " y Block" , " z Block" ,  " x Block Vel" , " y Block Vel" , " z Block Vel" , " 1x Wall" ," 1y Wall" ," 1z Wall" , " 2x Wall" ," 2y Wall" ," 2z Wall" , " 3x Wall" ," 3y Wall" ," 3z Wall" , " 4x Wall" ," 4y Wall" ," 4z Wall" , " 5x Wall" ," 5y Wall" ," 5z Wall" , " 6x Wall" ," 6y Wall" ," 6z Wall" , " 8x Wall" ," 8y Wall" ," 8z Wall" , " 9x Wall" ," 9y Wall" ," 9z Wall" ]
        
        #load imin/imax from training..  with some extensions. From code some of these values don't seem plausable (blockx for example) but we saw them in training data.  maybe nic mixed up some parms/files but won't hurt too much fi we mis some
        #fitwblpy output for initial state data


        # no point in computing probabilities if we won't use them in scoring, but we do use error type so let it go a bit longer
#        if(self.episode > (self.scoreforKL*4)): return 0;        

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
                                  4.5, 4.5, 4.5, #block vel  based on  programmed values.. though as it can drop with gravity on first step which we we don't see so 
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
        probv=0
        istate = self.format_istate_data(actual_state)
        # do base state for cart(6)  and pole (7)   looking at position and velocity. 
        for j in range (13):
            if(abs(istate[j]) > imax[j]):
                probv =  self.awcdf(istate[j],imax[j],iscale[j],ishape[j]);
                initprob += probv
                if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "& M42 LL1 " + "Step " + str(self.tick) +str(dimname[j]) + " init INCREASE  " + str(round(istate[j],3)) +" " + str(round(imax[j],3)) +" " + str(round(probv,3))+" " + str(round(iscale[j],3))
                    if(self.noveltyindicator != True) : self.logstr += "j=", str(j)+ "state = " + str(self.current_state)                    
               
            if(abs(istate[j]) < imin[j]):
                probv =  self.awcdf(abs(istate[j]),imin[j],iscale[j],ishape[j]);
                if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    initprob += probv
                    self.logstr +=  "& M42 LL1" + "Step " + str(self.tick) +str(dimname[j]) + " init DECREASE  " + str(round(istate[j],3)) +" " + str(round(imin[j],3)) +" " + str(round(iscale[j],3)) 
                    if(self.noveltyindicator != True) : self.logstr += "j=", str(j)+ "state = " + str(self.current_state)
            self.probvector[self.nextprob] = probv
            self.nextprob += 1                    

       ## look for walls to be in bad position 

        wallstart= len(istate) - 24
        if(wallstart < 19):  # whould have at least 2 blocks
                probv =  1
                initprob += probv
                if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "&M42 LL2  " + "Step " + str(self.tick) + "  BLOCK-QUANTITY-DECREASE (Level LL8 change) Len="+wallstart
        if(wallstart > 19+5*6):  # sould have at most 5 blocks
                probv =  1
                initprob += probv
                if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "&M42 LL2 " + "Step " + str(self.tick) + " BLOCK-QUANTITY-INCREASE (Level LL8 change) Len="+wallstart
        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    


       ## look for blocks to be in bad position or to have bad velocity
                    
        k=12 # where block data begins is 13 
        self.num_blocks = 0    
        for j in range (13,wallstart,1):
            probv=0
            k=k+1
            if(k==19):
                self.num_blocks += 1
                k=13;   #reset for next block
            # #if block too near wall then velocity seen at init might be a bounce so ignore it
            # if(k==16 and abs(istate[j-3]) >4): continue
            # if(k==17 and abs(istate[j-3]) >4): continue                
            # if(k==18 and abs(istate[j-3]) >9): continue
            # if(k==18 and abs(istate[j-3]) <1): continue            

            
            if(abs(istate[j]) > imax[k]):
                probv =  self.awcdf(abs(istate[j]),imax[k],iscale[k],ishape[k]);
                if((abs(istate[j]) - imax[k]) > 1) :
                    self.logstr +=  "& M42 LL2  " + "Step " + str(self.tick) + str(dimname[k])+ " init INCREASE " +  " " + str(round(istate[j],3)) +" " + str(round(imax[k],3)) +" " + str(round(probv,3)) +" " + str(round(iscale[j],3)) + " " + str(round(ishape[j],3))+" " + str(round(probv,3))
                    initprob += max(.24,probv)
#                    self.logstr += "j="+ str(j)+ str(self.current_state)                                        
                elif(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "& M42 LL2 " + "Step " + str(self.tick) + str(dimname[k])+ " init INCREASE " +  " " + str(round(istate[j],3)) +" " + str(round(imax[k],3)) +" " + str(round(probv,3)) +" " + str(round(iscale[j],3)) + " " + str(round(ishape[j],3))+" " + str(round(probv,3))
#                    self.logstr += str(self.current_state)                                        
            if(abs(istate[j]) < imin[k]):
                if( (imin[k] - abs(istate[j])) > 1 ):
                    self.logstr +=  "& M42 LL2 " + "Step " + str(self.tick) + str(dimname[k])+ " init DECREASE " +  " " + str(round(istate[j],3)) +" " + str(round(imax[k],3)) +" " + str(round(probv,3)) +" " + str(round(iscale[j],3)) + " " + str(round(ishape[j],3))+" " + str(round(probv,3))
#                    self.logstr += "j="+ str(j)+ str(self.current_state)                                        
                    initprob += max(.24,probv)
                else:
                    probv =  self.awcdf(abs(istate[j]),imin[k],iscale[k],ishape[k]);
                    initprob += probv
                    if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                        self.logstr +=  "&M42 LL2" + "Step " + str(self.tick) + " " + str(dimname[k]) + " init DECREASE " +  " " + str(round(istate[j],3)) +" " + str(round(imin[k],3)) +" " + str(round(probv,3)) +" " + str(round(iscale[j],3)) + " " + str(round(ishape[j],3))+" " + str(round(probv,3))
#                    self.logstr += "j="+ str(j)+ str(self.current_state)                                        
            self.probvector[self.nextprob] = probv
            self.nextprob += 1                    
        self.logstr  += ";"


       ## look for blocks to heading at cart's initital position

        probv=0                    
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k
#            dist = self.point_to_line_dist(istate[0:3],istate[k+nb*6:k+nb*6+2],istate[k+nb*6+3:k+nb*6+5])
            dist = self.point_to_line_dist(self.cart_pos(istate),
                                           self.block_pos(istate,nb),
                                           self.block_vel(istate,nb))
            probv=0                            
            if(dist < 1e-3): # should do wlb fit on this.. but for now just a hack as normal world data did not have enough data to fit
                probv = .5
            elif(dist < .01): # should do wlb fit on this.. but for now just a hack
                probv = (.01-dist)/(.01-1e-3)
                probv = .5*probv*probv   # square it so its a bit more concentrated and smoother                        
            initprob += probv

            if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                self.logstr +=  "&M42 LL3 " + "Step " + str(self.tick) +  " M42 Char Block " + str(nb) + " on initial direction attacking cart " +" with prob " + str(probv)
#            if(probv > .1):
#                self.uccscart.use_avoid_reaction=1            
#                self.logstr += str(self.current_state)

        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    

       ## look for blocks motions that heading to  other blocks initital position
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k                
#                dist = self.point_to_line_dist(istate[k+nb2*6:k+nb2*6+2],istate[k+nb*6:k+nb*6+2],istate[k+nb*6+3:k+nb*6+5])
                dist = self.point_to_line_dist(self.block_pos(istate,nb2),
                                               self.block_pos(istate,nb),
                                               self.block_vel(istate,nb))
                probv=0                                
                
                if(dist < 1e-3): # should do wlb fit on this.. but for now just a hack.  Note blocks frequently can randomly do this so don't consider it too much novelty.  Loose in test since they move before we see it
                    probv = .4            
                elif(dist < .01): # should do wlb fit on this.. but for now just a hack
                    probv = .4*(.01-dist)/(.01)
                initprob += probv
                if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "& M42 LL5" + "Step " + str(self.tick) +  " M42 Char Block " + str(nb) + " on initial direction aiming at block" + str(nb2) +" with prob " + str(probv)
#                    self.logstr += str(self.current_state)
        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    

                
       ## look for blocks motions that are parallel/or anti-parallel 
       
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                #  Use only block direction (velocity) for angle test  But if either vector norm is 0 then cannot compute angle, but in normal world block are never stationary so still abnormal
                angle = self.vector_angle(self.block_vel(istate,nb),self.block_vel(istate,nb2))
                # get weibul probabilities for the angles..  cannot be both small and large and weibul go to zero fast enough we
                probv=0                                
                if(angle < .1): probv = self.wcdf(angle,0.00,.512,.1218)
                if(angle >3.1): probv= self.rwcdf(angle,3.14,.512,.1218)
                if(probv > .5): probv= .5 #since this can happon randomly we never let it take longer                
                
                initprob += probv
                if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "& M42 LL3" + "Step " + str(self.tick) +  " Char Block motion " + str(nb) + " angle exception (e.g. parallel) to  block" + str(nb2) +  " with prob " + str(probv) + "for angle" + str(angle)
#                    self.logstr += str(self.current_state)                    
        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    



       ## look for blocks motions that are lines that will intersect 
       
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                dist = self.line_to_line_dist(self.block_pos(istate,nb),self.block_vel(istate,nb),self.block_pos(istate,nb2),self.block_vel(istate,nb2))
                # get weibul probabilities for the line-to-line-distance
                probv=0                
                if(dist < .025):
                    probw = self.wcdf(angle,0.013,.474,.136)
                    probv= min(.2,probw)  # this can occur randomly so limit its impact
                    initprob += probv
                    if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                        self.logstr +=  "& M42 LL3" + "Step " + str(self.tick) +  " M42 Char Block  " + str(nb) + " likely intersects with block" + str(nb2) +  " with probs " + str(probw) +" " + str(probv) + "for intersection distance" + str(dist)
#                        self.logstr += str(self.current_state)                    


        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    




        self.logstr  += ";"

        #        was limiting to one, but do want to allow more since  prob of novelty overall but we also discount this elesewhere
        if(initprob >self.maxinitprob):
            self.logstr += "Iprob clamped from" +  str(initprob)             
            initprob = self.maxinitprob


        self.probvector[self.nextprob] = initprob
        self.nextprob += 1                    

        if(initprob > self.minprob_consecutive):
             self.consecutiveinit = self.consecutiveinit+ 1
             if(self.consecutiveinit >3): self.initprobscale = min(self.initprobscale+.25,self.maxinitprob)
             if(self.uccscart.tbdebuglevel>1):             
                 print("Initprob cnt char ", initprob, self.cnt,self.logstr)
        else:
            self.consecutiveinit =0
        return initprob

    


    # get probability differene from checking state difference from prediction
    def cstate_diff_EVT_prob(self,cdiff,astate):
        dimname=[" x Cart" , " y Cart" , " z Cart" ,  " x Cart Vel" , " y Cart Vel" , " z Cart Vel ",  " x Pole" , " y Pole" , " z Pole" ," w Pole" ,  " x Pole Vel" , " y Pole Vel" , " z Pole Vel" , " z Block " , " y Block" , " z Block" ,  " x Block Vel" , " y Block Vel" , " z Block Vel" , " 1x Wall" ," 1y Wall" ," 1z Wall" , " 2x Wall" ," 2y Wall" ," 2z Wall" , " 3x Wall" ," 3y Wall" ," 3z Wall" , " 4x Wall" ," 4y Wall" ," 4z Wall" , " 5x Wall" ," 5y Wall" ," 5z Wall" , " 6x Wall" ," 6y Wall" ," 6z Wall" , " 8x Wall" ," 8y Wall" ," 8z Wall" , " 9x Wall" ," 9y Wall" ," 9z Wall" ] 


#        if(self.episode > (self.scoreforKL*4)): return 0;        

        if(self.dmax is None):        
        
            #load data from triningn
            self.dmax =    np.array([2.5664000e-02, 4.1890000e-02, 0.0000000e+00, #cart pos
                                     4.2490000e-02, 4.4770000e-02, 0.0000000e+00, #cart vel
                                     1.9012000e-02, 7.1500000e-03,8.4417000e-02, 2.3041000e-02,  #pole quat
                                     2.8452940e+00, 3.3939100e+00, 1.3531405e+01, #pole vel
                                     4.4438300e-01, 4.3333600e-01, 4.1041900e-01, #block pos
                                     2.22191570e-01, 2.26668060e-01, 2.05209520e-01]) #block vel Max  .. TB adjusted by had given how many errors in normal runs for phase 3 code..
#                                     1.4438300e-02, 1.3333600e-01, 1.1041900e-01, #block pos            
#                                     1.9191570e-01, 1.96668060e-01, 1.95209520e-01]) #block vel Max

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

        
        if(self.cnt < self.skipfirstNscores): return 0;    #need to be far enough along to get good prediciton

        prob=0  #where we accumualte probability
        probv=0
        charactermin=1e-2
        istate = cdiff
        # do base state for cart(6)  and pole (7) ..  because of noise we use only  a frac and ignore if around 1974 and if really large
        for j in range (13):
            if(self.uccscart.tbdebuglevel>0 and (istate[j] - imax[j]) >= 1.0):
                print( "Step " + str(self.tick) + dimname[j] + " ignored diff increase with state/max " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))                )
            if(istate[j] > imax[j] and   (istate[j] - imax[j]) < 1.0):
                probv =  self.cartprobscale*self.awcdf(abs(istate[j]),imax[j],iscale[j],ishape[j])
                #hack..    often the bug in system interface produces errors that have state around 1974.  Might skip a few real errors but should reduce false alarms a good bit
                if(abs(abs(istate[j])-.1974) < self.maxclampedprob/2 and len(self.logstr) < self.maxcarlen):
                    if(self.uccscart.tbdebuglevel>0):
                        self.logstr +=  "&" + "Step " + str(self.tick) + dimname[j] + " 1974 clampingbad  prob " + " " + str(round(probv,5)) + "  large bad state " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
                    probv=min(self.maxclampedprob,probv)
                else:
                    if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                        self.logstr +=  "&" + "M42 LL1 Step " + str(self.tick) + dimname[j] + " diff increase prob " + " " + str(round(probv,5)) + " s/l " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
            elif(istate[j] < imin[j] and  (imin[j] - istate[j] ) < 1 ):
                if(self.uccscart.tbdebuglevel>0 and (imin[j] - istate[j] ) >= 1 ):
                    print( "Step " + str(self.tick) + dimname[j] + " ignored diff too small with state/min " + str(round(istate[j],5)) +  " " + str(round(imin[j],5))                )
                probv =  self.cartprobscale*self.awcdf(abs(istate[j]),abs(imin[j]),iscale[j],ishape[j]);                
                #hack..    often the bug in system interface produces errors that have state around 1974.  Might skip a few real errors but should reduce false alarms a good bit
                if(abs(abs(istate[j])-.1974) < self.maxclampedprob/2 and len(self.logstr) < self.maxcarlen):
                    if(self.uccscart.tbdebuglevel>0):                    
                        self.logstr +=  "&" + "Step " + str(self.tick) + dimname[j] + " 1974 clampingbad   prob " + " " + str(round(probv,5)) + "  small bad state " + str(round(istate[j],5)) +  " " + str(round(imax[j],5))
                    probv=min(self.maxclampedprob,probv)
                else:
                    if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                        self.logstr +=  "& M42 LL1  Step " + str(self.tick) + dimname[j] + " diff decrease prob " + " " + str(round(probv,5)) + "  s/l " + str(round(istate[j],5)) +  " " + str(round(imin[j],5))
            self.probvector[self.nextprob] = probv
            self.nextprob += 1                    

#        prob += probv
        prob += .5*probv        #  too many funky errors causing detection on non-novel trials, so reduced this. 
#        print("at Step ", self.tick, " Dyn state ",istate)

#        if(self.episode > 20):        pdb.set_trace()


        probb=0
        #no walls in dtate diff just looping over blocks
        k=12 # for name max/ame indixing where we have only one block
        for j in range (13,len(istate),1):
            k=k+1
            if(k==19): k=13;   #reset for next block            
            #compute overall min/max for x,y position dimensions in actual state 
            if((k == 13 or k == 14) and astate[j] > self.blockmax):
                self.blockmax = astate[j]
            if((k == 13 or k == 14) and astate[j] < self.blockmin):
                self.blockmin = astate[j]
            if(k > 15 and k < 19):  # 16 17 and 18 are block velocity
                self.blockvelmax = max(astate[j],self.blockvelmax)
            
            # #if block are near boundaries where they bound position and velocity estimate can be  way off
            # if(k==13 and ((abs(astate[j]) >4)  or (abs(astate[j+1]) >4) or (abs(astate[j+2]) >8)or  (abs(astate[j+2]) <1)  )): continue
            # if(k==14 and ((abs(astate[j]) >4)  or (abs(astate[j-1]) >4) or (abs(astate[j+1]) >8)or  (abs(astate[j+1]) <1)  )): continue
            # if(k==15 and ((abs(astate[j]) >8)  or (abs(astate[j]) <1) or (abs(astate[j-2]) >4) or (abs(astate[j-1]) >4)    )): continue              
            # if(k==16 and ((abs(astate[j-3]) >4)  or (abs(astate[j+1-3]) >4) or (abs(astate[j+2-3]) >8)or  (abs(astate[j+2-3]) <1) )): continue
            # if(k==17 and ((abs(astate[j-3]) >4)  or (abs(astate[j-1-3]) >4) or (abs(astate[j+1-3]) >8)or  (abs(astate[j+1-3]) <1) )): continue
            # if(k==18 and ((abs(astate[j-3]) >8)  or (abs(astate[j-3]) <1) or (abs(astate[j-2-3]) >4) or (abs(astate[j-1-3]) >4)   )): continue              




            #probv=0
            #block motion not as predicted (domain independent test) but can be caused by may things and applies to L3, L5 and L7 as directions are off and  L4 since the bounce early produces a unepxcted position/velocity)
            #maybe some domain dependent stuff could differentiate
            # the random error (from block collisons with anything) sometimes cause large errors, so have to treat this a s very noisey and limit impact and only apply when resonable
            if(abs(istate[j]) > abs(imax[k]) and (abs(istate[j])- abs(imax[k]))<1):
                probb =  self.awcdf(abs(istate[j]),abs(imax[k]),iscale[k],ishape[k])  #  some randome error stll creap in so limit is impact below
                probv += probb
                if(istate[j] <0): dirstring=" DECREASE "
                else: dirstring=" INCREASE "
                if(probb>.05 and len(self.logstr) < self.maxcarlen):
                    self.logstr +=  "&M42 LL6 LL7 Block Motion Prediction Error" + "Step " + str(self.tick) + " " + str(dimname[k]) + dirstring + " prob of diff "  + str(round(probb,5)) + "  s/l " + str(round(istate[j],5)) +  " " + str(round(imax[k],5)) + " j-3is" +str(round(astate[j-3],3)) 
            elif (abs(istate[j]) > (abs(imax[k]))):   #Very large difference probably a collision with wall so just ignore it (and logging of error messag).
                probv += .001                
                if(len(self.logstr) < self.maxcarlen and self.uccscart.tbdebuglevel>1):
                    self.logstr +=  "&M42 block prediction error " + "Step " + str(self.tick) + " " + str(dimname[k]) + " diff way too large ignored prob.  s/l " + str(round(istate[j],5)) +  " " + str(round(imax[k],5)) + " j-3is" +str(round(astate[j-3],3)) 


            self.probvector[self.nextprob] = probv
            self.nextprob += 1                    
                    

        self.dynblocksprob += min(self.maxdynamicprob,probb)   # if we want to not impact of limit l3/l7 errors which can happen rnadomly and there are many block and many steps
                    


       ## look for blocks motions that heading to  other blocks position
        probv=0
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k                
                probv=0                                
                if(len(self.block_pos(astate,nb2)) == len(self.block_pos(astate,nb))):
                    dist = self.point_to_line_dist(self.block_pos(astate,nb2),
                                                   self.block_pos(astate,nb),
                                                   self.block_vel(astate,nb))


                    if(dist < 1e-3): # should do wlb fit on this.. but for now just a hack.  Note blocks frequently can randomly do this so don't consider it too much novelty
                        probv = self.maxdynamicprob            
                    elif(dist < .01): # should do wlb fit on this.. but for now just a hack
                        probv = self.maxdynamicprob * (.01-dist)/(.01)
                    prob += probv
                    if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                        self.logstr +=  "& M42 LL5" + "Step " + str(self.tick) +  " M42 Char Block " + str(nb) + " on diff  direction aiming at block" + str(nb2) +" with prob " + str(probv)
#                    self.logstr += str(self.current_state)

        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    
                
       ## look for blocks motions that are parallel/or anti-parallel 
       
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                #  Use only block direction (velocity) for angle test  But if either vector norm is 0 then cannot compute angle, but in normal world block are never stationary so still abnormal
                if(len(self.block_pos(astate,nb2)) == len(self.block_pos(astate,nb))):                
                    angle = self.vector_angle(self.block_vel(astate,nb),self.block_vel(astate,nb2))
                    # Cannot get weibul probabilities from training.. not enough data in normal world.
                    # But we can approxite with fake novelty and fill in by hand.  Have to be caeful with angle wrap 
                    deltav=0
                    if(angle < .02): deltav = .01 + self.wcdf(angle,0.00,.512,.1218)
                    if(angle >3.12): deltav = .01 + self.rwcdf(angle,3.14,.512,.1218)
                    if(deltav>.01 and len(self.logstr) < self.maxcarlen):
                        self.logstr +=  "& M42 LL3 or LL5" + "Step " + str(self.tick) +  " Char Block motion " + str(nb) + " dyn Block-Toward-Block " + str(nb2) +  " with probs " + str(deltav)  +" " + str(probv) + " for angle " +str(round(angle,3))
                    probv += deltav
        if(probv > self.maxdynamicprob): probv= self.maxdynamicprob             #since this can happon randomly we never let it take longer                
        prob += min(probv,self.maxdynamicprob)            

        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    

       ## look for blocks motions that are lines that will intersect 
        probv=0
        k=13 # where block data begins 
        for nb in range(self.num_blocks):
            for nb2 in range(nb+1,self.num_blocks):
                if(len(self.block_pos(astate,nb2)) == len(self.block_pos(astate,nb))):                                
                    dist = self.line_to_line_dist(self.block_pos(astate,nb),self.block_vel(astate,nb),self.block_pos(astate,nb2),self.block_vel(astate,nb2))
                    # get weibul probabilities for the line-to-line-distance
                    probv=0                
                    if(dist < .025):
                        probw = .01+self.wcdf(angle,0.013,.474,.136)
                        probv+= probw  # this can occur randomly so limit its impact
                        if(probw>charactermin and len(self.logstr) < self.maxcarlen):
                            self.logstr +=  "& M42 LL3 or LL5" + "Step " + str(self.tick) +  " M42 Char Block  " + str(nb) + " dyn likely intersects with block" + str(nb2) +  " with probs " + str(probw) +" " + str(probv) + "for intersection distance" + str(dist)

        prob += min(probv,self.maxdynamicprob)            
        self.logstr  += ";"
        self.probvector[self.nextprob] = probv
        self.nextprob += 1                    

        prob=min(self.maxdynamicprob,prob)        

        if(prob > 1): prob = 1


        return prob




        # get probability differene froom initial state
    def istate_diff_G_prob(self,actual_state):
        dimname=[" x Cart" , " y Cart" , " z Cart" ,  " x Cart Vel" , " y Cart Vel" , " z Cart Vel ",  " x Pole" , " y Pole" , " z Pole" ," w Pole" ,  " x Pole Vel" , " y Pole Vel" , " z Pole Vel" , " z Block " , " y Block" , " z Block" ,  " x Block Vel" , " y Block Vel" , " z Block Vel" , " 1x Wall" ," 1y Wall" ," 1z Wall" , " 2x Wall" ," 2y Wall" ," 2z Wall" , " 3x Wall" ," 3y Wall" ," 3z Wall" , " 4x Wall" ," 4y Wall" ," 4z Wall" , " 5x Wall" ," 5y Wall" ," 5z Wall" , " 6x Wall" ," 6y Wall" ," 6z Wall" , " 8x Wall" ," 8y Wall" ," 8z Wall" , " 9x Wall" ," 9y Wall" ," 9z Wall" ] 
        
        #load mean/std from training..  
        #if first time load up data.. 
        if(self.imean is None):

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
            if(probv>charactermin and len(self.logstr) < self.maxcarlen):
#                initprob += probv
                initprob = max(initprob,probv)                
                self.logstr +=  "&" + "Step " + str(self.tick) +  str(dimname[j]) + " init out of range  " + str(round(istate[j],3)) +" " + str(round(imean[j],3)) +" "  + str(round(istd[j],3)) +" " + str(round(probv,3))

        wallstart= len(istate) - 24                    
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,wallstart,1):
            if("Wall"  in str(dimname[j])): break
            probv =  self.gcdf(istate[j],imean[j],istd[j]);
            if(probv>charactermin and len(self.logstr) < self.maxcarlen):
#                initprob += probv
                initprob = max(initprob,probv)                
                self.logstr +=  "&" + "Step " + str(self.tick) +  str(dimname[j]) + "  init out of range  " + str(round(istate[j],3)) +" " + str(round(imean[j],3)) +" "  + str(round(istd[j],3)) +" " + str(round(probv,3))
            k = k +1
            if(k==19): k=13;   #reset for next block
        self.logstr  += ";"

        if(initprob >1): initprob = 1

        if(initprob > self.minprob_consecutive):
             self.consecutiveinit = min(self.consecutiveinit+ 1,self.maxconsecutivefailthresh)
             if(self.uccscart.tbdebuglevel>1):             
                 print("Initprob cnt char ", initprob, self.cnt,self.logstr)
        else:
            self.consecutiveinit =0
        return initprob

    


    # get probability differene froom continuing state difference
    def cstate_diff_G_prob(self,cdiff):
        dimname=[" x Cart" , " y Cart" , " z Cart" ,  " x Cart Vel" , " y Cart Vel" , " z Cart Vel ",  " x Pole" , " y Pole" , " z Pole" ," w Pole" ,  " x Pole Vel" , " y Pole Vel" , " z Pole Vel" , " z Block " , " y Block" , " z Block" ,  " x Block Vel" , " y Block Vel" , " z Block Vel" , " 1x Wall" ," 1y Wall" ," 1z Wall" , " 2x Wall" ," 2y Wall" ," 2z Wall" , " 3x Wall" ," 3y Wall" ," 3z Wall" , " 4x Wall" ," 4y Wall" ," 4z Wall" , " 5x Wall" ," 5y Wall" ," 5z Wall" , " 6x Wall" ," 6y Wall" ," 6z Wall" , " 8x Wall" ," 8y Wall" ," 8z Wall" , " 9x Wall" ," 9y Wall" ," 9z Wall" ] 

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
            if(probv>charactermin and len(self.logstr) < self.maxcarlen):
#                prob += probv
                prob = max(prob,probv)                
                self.logstr +=  "&" + "Step " + str(self.tick) +  str(dimname[j]) + " dyn out of range  " + str(round(istate[j],6)) +" " + str(round(imean[j],6)) +" "  + str(round(istd[j],6)) +" " + str(round(probv,3))

        

        #no walls in dtate diff just looping over blocks
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,len(istate),1):
            if("Wall"  in str(dimname[j])): break
            probv =  self.gcdf(istate[j],imean[j],istd[j]);
            if(probv>charactermin and len(self.logstr) < self.maxcarlen):
                prob += probv
                prob = max(prob,probv)                
                self.logstr +=  "&" + "Step " + str(self.tick) +  str(dimname[j]) + " dyn out of range  " + str(round(istate[j],6)) +" " + str(round(imean[j],6)) +" "  + str(round(istd[j],6)) +" " + str(round(probv,3))
            k = k +1
            if(k==19): k=13;   #reset for next block
        self.logstr  += ";"                

        if(prob > self.minprob_consecutive):
            if(prob > 1): prob = 1

        return prob


    


    def world_change_prob(self,settrain=False):

        # don't let first episodes  impact world change.. need stabilsied scores/probabilites.. skipping work here also makes it faster
        # if(self.episode< 0*self.scoreforKL):
        #     self.worldchangedacc = 0
        #     self.worldchangeblend = 0
        #     self.previous_wc = 0                       
        #     return self.worldchangedacc            

            

        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1193"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1):             
                print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc

        self.summary ="" #reset summary to blank
            
        

        
        mlength = len(self.problist)
        mlength = min(self.scoreforKL,mlength)
        # we look at the larger of the begging or end of list.. world changes most obvious at the ends. 

        window_width=7
        #look at list of performacne to see if its deviation from training is so that is.. skip more since it needs to be stable for window smoothing+ mean/variance computaiton
        PerfKL =    0
        pmu=0
        psigma=0        
        if (len(self.perflist) >(self.scoreforKL+window_width) and len(self.perflist) < 3* self.scoreforKL ):  
            #get smoothed performance 
            cumsum_vec = np.cumsum(np.insert(self.perflist, 0, 0))
            smoothed = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
            pmu = np.mean(smoothed[:-window_width])  # we skip first/iniiprob... it is used elsehwere. 
            psigma = np.std(smoothed[:-window_width])
            
#            if(pmu <  self.mean_perf or pmu >  self.mean_perf +  self.stdev_perf):     #if we want only  KL for those what have worse performance or much better                
            if(pmu <  self.mean_perf ):     #if we want only  KL for those what have worse performance 
                #use model stdev since KL will see 0 stdev as different even if its actually just a good thing                
                PerfKL = self.kullback_leibler(pmu, psigma, self.mean_perf, self.stdev_perf)  
                self.debugstring = '   PerfKL {} {} {} {} PerfKL={} perlist={}= ,'.format(pmu, psigma, self.mean_perf, self.stdev_perf, round(PerfKL,3),self.perflist)
                if(self.uccscart.tbdebuglevel>1):             
                    print(self.debugstring)
            elif(pmu >  (self.mean_perf+.05) and self.episode > self.scoreforKL and self.episode < 2* self.scoreforKL  and (self.levelcnt[6] + self.levelcnt[7]) >  3 * (self.episode-self.scoreforKL)):     #if we we are really much better and we are seeing potential novelty,  report it
                #use model stdev since KL will see 0 stdev as different even if its actually just a good thing                
                PerfKL = self.kullback_leibler(pmu, psigma, self.mean_perf+.05, self.stdev_perf)  # if really much better
                self.debugstring = '   E{} {} BetterPerfKL {} {} {} {} PerfKL={} perlist={}'.format(self.episode, (self.levelcnt[6]+self.levelcnt[7])/(self.episode-20),  pmu, psigma, self.mean_perf, self.stdev_perf, round(PerfKL,3),self.perflist)
                if(len(str(self.hint))>15): self.debugstring +=  self.hint[9:15]
                if(self.uccscart.tbdebuglevel>-1):             
                    print(self.debugstring)
                

                
            # If there is still too much variation (too many FP) in the variance in the small window so we use stdev and just new mean this allows smaller (faster) window for detection. 
            # PerfKL = self.kullback_leibler(pmu, self.stdev_perf, self.mean_perf, self.stdev_perf)
        else:
            PerfKL = 0

        
        if( self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1222"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1):             
                print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            if(self.previous_wc > .5): self.worldchangedacc=self.previous_wc

        

        if(mlength > 1) :
            mu = np.mean(self.problist[0:mlength-1])
            sigma = np.std(self.problist[0:mlength-1])
        else:
            mu = sigma = 0
            self.debugstring = '   ***Zero Lenth World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,5),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
            if(self.uccscart.tbdebuglevel>1):             
                print(self.debugstring)
            self.logstr +=  "&" + self.debugstring
            
            self.worldchanged = self.worldchangedacc
            return max(self.worldchangedacc,self.previous_wc);
        
       
        if(settrain):
           self.mean_train = mu;
           self.stdev_train = sigma;
           if(self.uccscart.tbdebuglevel>1):             
               print("Set  world change train mu and sigma", mu, sigma)
           self.logstr +=  "&" + "Set  world change train mu and sigma" + str(mu) + str(sigma) +" saying world_change = 0"
           self.worldchanged = 0
           return max(self.worldchangedacc,self.previous_wc);
        
        if( self.mean_train == 0):
            self.mean_train = 0.004
            self.stdev_train = 0.009
            self.dynam_prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.

        if(mu > self.mean_train):
            self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)
        else:
            self.KL_val=0            
        self.debugstring = '   ***Short World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
            round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
            round(self.stdev_train,3) ,round(self.KL_val,5), round(self.KL_threshold,5), "\n", [round(num,2) for num in self.scorelist])
        if (self.debug):
            print(self.debugstring)
           
        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1260"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1):             
                print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc


        dprob = perfprob = 0                # don't allow on short runs.. dynamics and performance are off            

        if (len(self.problist) < 198):   #for real work but short list
            self.consecutivesuccess=0            
            self.failcnt += 1
            if( self.consecutivefail >0):
                self.consecutivefail = min(self.consecutivefail+1, self.maxconsecutivefailthresh+2)
                if(self.consecutivefail > self.maxconsecutivefail):
                    self.maxconsecutivefail = self.consecutivefail
                    if(self.maxconsecutivefail > self.maxconsecutivefailthresh):
                        self.worldchangedacc = 1
                        self.logstr +=  "&" + "Step " + str(self.tick) + "#####? Uncontrollable world -- too many consecutive failures.  Guessing actions were remapped/perturbed but will take a while to confirm ##### "                         

            else: self.consecutivefail=1
            if(mu > self.mean_train):
                self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)
            else:
                self.KL_val=0                
                
            self.debugstring = '   ***Short World Change Acc={}, Failcnt= {} Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
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
                    self.logstr +=  "&" + self.debugstring
                    return max(self.worldchangedacc,self.previous_wc);

            else:
                sigma = self.stdev_train


            if(mu < self.mean_train):   #no point computing if world differences are smaller, it may be "much" smaller but that is okay
                self.KL_val = 0   
            else: 
                self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)


            #KLscale = (self.num_epochs + 1 - self.episode / 2) / self.num_epochs  # decrease scale (increase sensitvity)  from start 1 down to  1/2
            #        KLscale = min(1, 4*(1 + self.episode) / num_epochs)  # decrease scale (increase sensitvity)  from start 1 down to  1/2
            KLscale = 1
            dprob = min(1.0, ((KLscale * self.KL_val) )) 
            perfprob = min(1.0, self.PerfScale * PerfKL)  #make this smaller since it is slowly varying and  added every time.. less sensitive (good for FP avoid, sloer l


        #random collisions can occur and they destroy probability computation so ignore short long length if they occur, but nor if being attacked or failing a lot
        if( (self.logstr.count("CP") > 1)  and mlength < (self.scoreforKL+2) and ( "attack" not in self.logstr) and (self.consecutivefail < 2)):
            dprob=.0009990
            prob = min(1,max(dprob,perfprob)) # use max of dynamic and long-term performance probabilities.
            if(self.uccscart.tbdebuglevel>1): print("Debug found ", self.logstr.count("CP"), "CPs in string")
        else:
            prob = min(1,max(dprob,perfprob)) # use max of dynamic and long-term performance probabilities.
            if(self.uccscart.tbdebuglevel>1): print("Debug did not find many CP ", self.logstr.count("CP"), " in string. dprob,perfprob = ",
                  str(dprob), " ", str(perfprob), " ", str(self.KL_val), "Prob=",str(prob))


        self.levelcnt=np.zeros(10)
        i=0
        lsum=0                        
        i+= 1; self.levelcnt[i] =L1= self.trialchar.count("LL1"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L2block= self.trialchar.count("LL2"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L3block= self.trialchar.count("LL3"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L4block= self.trialchar.count("LL4"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L5block= self.trialchar.count("LL5"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L6block= self.trialchar.count("LL6"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L7block= self.trialchar.count("LL7"); lsum += self.levelcnt[i] 
        i+= 1; self.levelcnt[i] =L8block= self.trialchar.count("LL8")*10; lsum += self.levelcnt[i] 
        
        scale = (self.episode-self.scoreforKL)
        if(scale >0 ):
            lprob=0
            psum =  .05 * self.rwcdf(lsum*scale,2.24285, 1.123764, .58301)             #weibul probabiliy based on scaled sum, but its often true for non-novel and is done on every episode so weight it small so this alone cannot get to .5
            lprob =  min(1,prob+psum);
            if(self.uccscart.tbdebuglevel>-1 and scale < 1  and len(str(self.hint))>16):
                print('dEp {}.{} detsum= {} ssum= {} psum {} WC {} lprob {} , hint=|{}|'.format(self.episode,self.tick,lsum, lsum*scale,  psum, self.worldchangedacc,lprob,str(self.hint)[9:15]))
            if(self.episode > self. scoreforKL and self.episode < 2*self. scoreforKL): prob = lprob
                


            

        #if we had  collisons and not consecuretive faliures, we don't use this episode for dynamic probability .. collisions are not well predicted
        #tlen = min(self.scoreforKL,len(self.uccscart.char))
                   
        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1325"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1): print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc
            

        if (len(self.problist) < self.scoreforKL):
            if(prob < .9):    
                self.worldchanged = prob * len(self.problist)/(self.scoreforKL)
                if(self.uccscart.tbdebuglevel>1):                print("worldchange from problist <.9",self.worldchanged)
            else:
                self.worldchanged = prob   #Phase3   addition  to let it be faster with detection when it has really high probability
                if(self.uccscart.tbdebuglevel>1):                print("worldchange from problist >9",self.worldchanged)                
        elif (len(self.problist) < 2* self.scoreforKL):
            self.worldchanged = prob            
        else: # if very long list, KL beceomes too long, its more likely to be higher from random agent crashing into pole so we the impact
            self.worldchanged = prob * (2*self.scoreforKL)/len(self.problist)
            if(self.uccscart.tbdebuglevel>1):                print("worldchange from long problist",self.worldchanged)                            
            

            
        #only do blockmin/max if we did a long enough  note this does not need any blending or scoreforKL since its an absolute novelty to have this happen
        # if the blockmax/min (in general from past episode) from past are not normal add them to the worldchange acce.  Again should be EVT based but not enough training yet.
        if (self.episode > (self.scoreforKL) and self.tick == 190 and self.blockvelmax > 4):   # if we have been through enough episodes and enough steps and blocks moved enough
            minmaxupdate = min(abs(self.blockmin-self.normblockmin),abs(self.blockmax-self.normblockmax))
            if(minmaxupdate > .2):
                if(self.blockmin < self.normblockmin or self.blockmax> self.normblockmax):
                    self.logstr += '&& M42 LL4 BLock SHRINK Size decreased. minmaxupdate='+ str(minmaxupdate) 
                elif(self.blockmin > self.normblockmin or self.blockmax< self.normblockmax):
                    self.logstr += '&& M42 LL4 BLock GROW Size increased ' + str(minmaxupdate)
                if(self.uccscart.tbdebuglevel>1): print('  minmaxupdate {} min {} max {} normal {} {}  ,'.format(minmaxupdate,self.blockmin,self.blockmax,self.normblockmin,self.normblockmax)        )
                prob = prob+ minmaxupdate/4  #dont add too much as it can be noisy
#                self.worldchanged += .3              
                self.debugstring = '   Prob {} after Level LL4 Size Change minmaxupdate {} min {} max {} normal {} {} VelMax {}  ,'.format(prob,minmaxupdate,self.blockmin,self.blockmax,self.normblockmin,self.normblockmax,self.blockvelmax)
                if(self.uccscart.tbdebuglevel>1): print(self.debugstring)
                self.logstr +=  "&" + self.debugstring                


        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1314"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1): print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc                              


        outputstats=False
            

#####!!!!!##### end GLue CODE for EVT

#####!!!!!#####  Domain Independent code

        failinc = 0
        #if we are beyond KL window all we do is watch for failures to decide if we world is changed

            
        if(self.episode > self.scoreforKL+1 and self.episode < 2* self.scoreforKL):
            #pdb.set_trace()
            if(self.failcnt/(self.episode+1) > self.maxfailfrac):
                self.worldchanged = 1
                if(self.uccscart.tbdebuglevel>1):                print("worldchange from  failcnt",self.worldchanged)                                
                self.logstr +=  "&" + "Step " + str(self.tick) + "World Change detected by Very High FailFrac=" + str(self.failcnt/(self.episode+1))           
                faildiff=0                
            else: 
                faildiff = self.failcnt/(self.episode+1)-self.failfrac            
            if(faildiff > 0):
                self.logstr +=  "&" + "Step " + str(self.tick) + "High FailFrac=" + str(self.failcnt/(self.episode+1))
                failinc = max(0,  ((faildiff)*self.failscale)) 
                failinc *= min(1,(self.episode - self.scoreforKL)/self.scoreforKL)   #Ramp it up slowly as its more unstable when it first starts at scoreforKL
                failinc = min(1,failinc)

            #world change blend  can go up or down depending on how probablites vary.. goes does allows us to ignore spikes from uncommon events. as the bump i tup but eventually go down. 
            if(prob < .5 and  self.worldchangedacc <.5) : # blend wo  i.e. decrease world change accumulator to limit impact of randome events
                #self.worldchangeblend = min(self.worldchangedacc * self.blenddownrate, (self.blenddownrate *self.worldchanged + (1-self.blenddownrate) * self.worldchangeblend ))
                self.worldchangedacc = min(1,self.worldchangedacc*self.worldchangeblend)            
                self.debugstring = "BlendDown "                

            else:
                #worldchange acc once its above .5 it cannot not go down.. it includes max of old value..
                self.worldchangeblend = min(1, (        self.blenduprate *self.worldchanged + (1-self.blenduprate) * self.worldchangeblend ))

                self.debugstring = "Blendup using rate " + str(self.blenduprate) + "wc/wcb="+ str(self.worldchanged) + " " + str(self.worldchangeblend)                                
                # we add in an impusle each step if the first step had initial world change.. so that accumulates over time

                if(len(self.problist) > 0 ) :
                    self.worldchangedacc = min(1,self.problist[0]*self.initprobscale + (self.worldchangedacc+self.worldchanged)*min(1,self.worldchangeblend+failinc))
                else:
                    self.worldchangedacc = min(1,max(self.worldchangedacc+self.worldchanged,self.worldchangedacc*self.worldchangeblend+failinc))
                self.debugstring += '    mu={}, sig {}, mean {} stdev{}  WCacc WBlend {} {} vals {} {} {} thresh {} '.format(
                    round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3), round(self.worldchangedacc,3), round(self.worldchangeblend,3) ,round(self.KL_val,3),round(PerfKL,3), dprob,  "\n")
                if(self.uccscart.tbdebuglevel>1): print(self.debugstring)


        if( self.previous_wc > self.worldchangedacc and self.previous_wc>.1 ):
            self.debugstring = "   worldchanged when down, line 1358 "+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1): print(self.debugstring)            
            #self.worldchangedacc=self.previous_wc          

        # Until we have enough data, reset and worldchange, don't start accumulating
        if(self.episode < self.scoreforKL):
            self.worldchangedacc = 0
            self.worldchangeblend = 0            

            # normal world gets randome incremenets not bigger ones all in a row..  novel world can get many consecutitive one so we increase prob of that big an increment and expodentially based on sequence length and .
            # Parms weak as not much data in training as its too infrequently used.  Not used later in stage as random errors seem to grow with resets so consecurive random becomes more likely
        if((self.worldchangedacc - self.previous_wc) > self.minprob_consecutive and self.episode < 2* self.scoreforKL):
            if(self.consecutivewc >0):
                wcconsecutivegrowth = (self.consecutivewc/20)*  (1-self.wcdf((self.worldchangedacc - self.previous_wc),.001,.502,.13))
                if(self.uccscart.tbdebuglevel>1): print("wcconsecutivegrowth = ", wcconsecutivegrowth, self.worldchangedacc, self.previous_wc)
                self.logstr += 'World Change Consecutive {}, prev {} base {} increent {} '.format(self.consecutivewc,round(self.previous_wc,3), round(self.worldchangedacc,3),round(wcconsecutivegrowth,3))
                self.worldchangedacc = min(1, self.worldchangedacc  + wcconsecutivegrowth)
                if(self.uccscart.tbdebuglevel>1): print("EPi2 previs new world change", self.episode, self.previous_wc, self.worldchangedacc)
                
            self.consecutivewc += 1
        else:  self.consecutivewc = 0

        
#####!!!!!#####  End Domain Independent code tor consecurtiv efailures


#####!!!!!#####  Start API code tor reporting
        self.logstr += 'World Change Acc={} {} {} {}, CW={},CD={} D/KL Probs={},{}'.format(round(self.worldchangedacc,3), round(self.worldchangeblend,3),round(self.previous_wc,3),round(failinc,3), self.consecutivewc,self.dynamiccount,round(dprob,3), round(perfprob,3))
        


        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1389"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1): print(self.debugstring)            


        if(self.uccscart.tbdebuglevel>1): print("EPi previs new world change", self.episode, self.previous_wc, self.worldchangedacc)
        if(self.previous_wc < .5 and self.worldchangedacc        >= .5):
            if(self.noveltyindicator == True):
                self.logstr +=  self.hint                
                self.logstr += "#!#!#!  World change TP Detection "+ str(self.episode) + "  @@ FV= "
                self.summary += "#!#!#!  World change TP Detection "+ str(self.episode) + "  @@  "                
            elif(self.noveltyindicator == False):
                self.logstr +=  self.hint
                self.logstr += "#!#!#!  World change FP Detection "+ str(self.episode) + "  @@ FV= "
                self.summary += "#!#!#!  World change FP Detection "+ str(self.episode) + "  @@  "                                
            else: self.logstr += "#!#!#!  World change blind Detection "+ str(self.episode) + "  @@ FV= "    
            self.logstr += str(self.current_state)
            print ("   Detecting at Episode==", self.episode, " hint=",self.hint); 
            outputstats=True            

        # if world changed an dour performance is below .65  we start using avoidance reaction
        #        if(self.worldchangedacc        >= .6 and (100*self.perf/self.totalcnt) < 65) :
        #                self.uccscart.use_avoid_reaction=True            


        outputstats=True  #alays true now that they want stats per episode
        finalepisode =  False
        if((self.episode+1)%99 ==0 ): finalepisode =  True
        if(finalepisode or outputstats):
            initcnt = self.trialchar.count("init")
            blockcnt = self.trialchar.count("Block")
            blockvelcnt = self.trialchar.count("Block Vel")
            blockmotion = self.trialchar.count("Block Motion")                            
            polecnt = self.trialchar.count("Pole")
            cartcnt = self.trialchar.count("Cart")
            smallcnt = self.trialchar.count("small")
            inccnt = self.trialchar.count("INCREASE")
            deccnt = self.trialchar.count("DECREASE")
            shrinkcnt = self.trialchar.count("SHRINK")
            growcnt = self.trialchar.count("GROW")                        
            largecnt = self.trialchar.count("large")                                
            diffcnt = self.trialchar.count("diff")
            velcnt = self.trialchar.count("Vel")                                
            failcnt = self.trialchar.count("High")
            attcart = self.trialchar.count("Relational Attacking cart")
            aimblock = self.trialchar.count("Relational aiming")
            parallelblock = self.trialchar.count("Relational parallel")
            
            #initalize characterization
            self.uccscart.characterization['entity']=None
            self.uccscart.characterization['attribute']=None
            self.uccscart.characterization['change']=None           
            

            # if we get enough L5 we increase our world change estimate
            if(L5block > 50 and self.episode < 2*self.scoreforKL):
                self.worldchangedacc = min(1,.25+self.worldchangedacc);
                
            maxi = np.argmax(self.levelcnt)
            
            # level 1 very noisy.. if its the max
            if(maxi == 1 and self.levelcnt[1] < 1000):
                self.levelcnt[1] = self.levelcnt[1] / 100
            maxi = np.argmax(self.levelcnt)                



            
            if(maxi == 6 or maxi == 7  and  abs(self.levelcnt[6]-self.levelcnt[7]) < 5):  #6 and 7 often confused  if nearly equal then  do another test
                incdec_ratio =  min(inccnt,deccnt)/(1+max(inccnt,deccnt))
                #level 6 tends to be more biased so ratio is < .3 (generally < .2)  while 7 is > .4 
                if(incdec_ratio < .3):
                    self.levelcnt[6] += 10
                else:
                    self.levelcnt[7] += 10                    
                maxi = np.argmax(self.levelcnt)                

            if(self.uccscart.characterization['level'] == int(8)):
                maxi=8
                self.levelcnt[8] += 10000                                    


            #fil in M42    Characterization with proabiltiies based on level counts
            levelprobs = np.zeros(9)
            self.levelcnt[0] = 0
            rescale = 1.0/(np.sum(self.levelcnt)+(1-self.worldchangedacc)*100)
            # before we are computing scores its very noisy so just ingore so it does not impact accumated data            
            if(self.episode < self.scoreforKL):
                rescale = 0
                #self.trialchar=""                
            levelprobs = rescale * self.levelcnt
            levelprobs[0] = 1-np.sum(levelprobs)
            #m42 update after seeing we got too many errors on level 0 (no novelty)
            if(self.worldchangedacc < .5):
                levelprobs[0] = max(levelprobs) + .5 - self.worldchangedacc
                levelprobs = levelprobs/ sum(levelprobs)  #renormalize

            ljson = [dict() for x in range(9)]
            for i in range(9):
                if(levelprobs[i]>0):
                    ljson[i] = {"level_number":i, "Prob":round(levelprobs[i],2)}
            
            self.uccscart.characterization['level']=ljson
                
                

            # if level is 8 is already filled in,  no need to do any filling
            if(maxi ==8):
                #8 had special code to do increase vs increasing
                self.uccscart.characterization['entity']="Block"; 
                self.uccscart.characterization['attribute']="quantity";
                if(self.logstr.count("LL8: Blocks quantity dec") > self.logstr.count("LL8: Blocks quantity inc")  ):  #if we had more than one chance its increasing
                    if(self.logstr.count("LL8: Blocks quantity dec") >3  ):  #if we had more than one chance its increasing                    
                        self.uccscart.characterization['change']='decreasing';
                    else:
                        self.uccscart.characterization['change']='decrease';                        
                else:
                    if(self.logstr.count("LL8: Blocks quantity increaseing") >3  ):  #if we had more than one chance its increasing                                        
                        self.uccscart.characterization['change']='increasing';
                    else:
                        self.uccscart.characterization['change']='increase';                        
            else:
                self.uccscart.characterization['entity']=None
                self.uccscart.characterization['attribute']=None
                self.uccscart.characterization['change']=None           
                
                if(maxi == 1 and self.levelcnt[1] > 1000):
                    if(cartcnt > polecnt):
                        self.uccscart.characterization['entity']="Cart";
                    else:
                        self.uccscart.characterization['entity']="Pole";
                    self.uccscart.characterization['attribute']="speed";
                    if(inccnt > deccnt):
                        self.uccscart.characterization['change']='increase';
                    else:
                        self.uccscart.characterization['change']='decrease';                    
                else:

                    self.levelcnt[1] = self.levelcnt[1]/1000 # reduce  level  as its noisy and often large but when really there its 1000s ao if here even if its the max something else is goign one. 
                    maxi = np.argmax(self.levelcnt)
                    if(self.levelcnt[maxi]==0):
                        maxi=-1
                    else:
                        self.uccscart.characterization['entity']="Block";
                            
                    if (maxi == 7  or velcnt > 10*L4block):    #l7 will have huge numbers of blockvelocity violations
                        # should do more to figure out direction of 
                        self.uccscart.characterization['attribute']="direction";
                        if(attcart > 5):
                           self.uccscart.characterization['change']='toward cart';                           
                        elif(aimblock > 5 or  parallelblock > 5 ):
                            self.uccscart.characterization['change']='toward block';
                        else:
                            self.uccscart.characterization['change']='toward location';
                    
                    if(maxi == 3 or  (maxi == 2 and (initcnt < diffcnt) )):
                        maxi =3
                        self.uccscart.characterization['attribute']="direction";
                        self.uccscart.characterization['change']='toward location';
                        if(attcart > 5):
                           self.uccscart.characterization['change']='toward cart';                                                   
                    if (maxi == 4 or L4block> max(0,(self.episode-self.scoreforKL)/2)):
                        #level 4  if most episodes since we started scoring  showin level 4 we stick with 4.. L5 gets to count every tick, l4 only toward end
                        maxi=4 #reset to 5 does not overwrite
                        self.uccscart.characterization['attribute']="size";
                        if(shrinkcnt < growcnt):
                            self.uccscart.characterization['change']='increase';
                        else: 
                            self.uccscart.characterization['change']='decrease';
                        if( max(shrinkcnt,growcnt) > 400 and shrinkcnt < growcnt):
                            self.uccscart.characterization['change']='increasing';
                        else: 
                            self.uccscart.characterization['change']='decreasing';

                            
                    elif (maxi == 2):
                        self.uccscart.characterization['attribute']="speed";
                        if(inccnt > deccnt):
                            self.uccscart.characterization['change']='increase';
                        else:
                            self.uccscart.characterization['change']='decrease';                    
                    elif (maxi == 5):
                        self.uccscart.characterization['attribute']="direction";
                        if(attcart > 5):
                           self.uccscart.characterization['change']='toward cart';                           
                        elif(aimblock > 5 or  parallelblock > 5 ):
                            self.uccscart.characterization['change']='toward block';
                        else:
                            self.uccscart.characterization['change']='toward location';
                    elif(maxi == 6):
                        # should do more to figure out direction impact...  maybe use direction if falling tooo fast or
                        # slow.  ALso friction could be here..
                        self.uccscart.characterization['attribute']="gravity";
                        if(inccnt > deccnt):
                            self.uccscart.characterization['change']='increase';
                        else:
                            self.uccscart.characterization['change']='decrease';




            #if world changed don't declare a max
            if(self.worldchangedacc < .5):
                maxi = 0
                if(self.worldchangedacc < .3):                # if very sure its not-novel remove the characterization
                    #m42 update for errors when non-novel
                    self.uccscart.characterization['change']=None;                
                    self.uccscart.characterization['attribute']=None;                                
                            



            
                
            self.trialchar += self.logstr  #save char string without any added summarization so we can compute over it. 

            if(not finalepisode):
                self.summary += "@@@@@ Episode Characterization for  world change prob: " +str(self.worldchangedacc) +" "            
            else:
                #if final, truncate the characterstring so its just the final data
                if(self.worldchangedacc        <.5):
                    self.summary += "##### @@@@@ Ending Characterization of potential observed novelities, but did not declare world novel  with  world change prob: " +str(self.worldchangedacc)
                    if(self.worldchangedacc        >= .5):                
                        self.summary += "##### @@@@@  Ending Characterization of observed novelities in novel world   with  world change prob: " +str(self.worldchangedacc)
            if(initcnt > diffcnt ):
                self.summary += "Inital world off and "
            if(diffcnt > initcnt ):
                self.summary += " Dynamics of world off and "                    
            if(blockcnt > polecnt and blockcnt > cartcnt ):                    
                self.summary += " Dominated by Blocks with"
            if(cartcnt > polecnt and cartcnt >  blockcnt   ):                    
                self.summary += " Dominated by Cart with"
            if(polecnt > cartcnt and polecnt > blockcnt ):                    
                self.summary += " Dominated by Pole with"
            self.summary += " Velocity Violations " + str(velcnt)                                                                                                
            self.summary += "; Agent Velocity Violations " + str(blockvelcnt)                
            self.summary += "; Cart Total Violations " + str(cartcnt)
            self.summary += "; Pole Total Violations " + str(polecnt)
            self.summary += "; Speed/position decrease Violations " + str(deccnt)
            self.summary += "; Speed/position increase Violations " + str(inccnt)                                                                                                            
            self.summary += "; Attacking Cart Violations " + str(attcart)
            self.summary += "; Blocks aiming at blocks " + str(aimblock)                                                                                                            
            self.summary += "; Coordinated block motion " + str(parallelblock)
            self.summary += "; Agent Total Violations " + str(blockcnt + parallelblock + attcart + blockvelcnt)
            for i in range(0,9):
                self.summary += "; L" + str(i) + ":=" + str(self.levelcnt[i])
            self.summary += ";  Violations means that aspect of model had high accumulated EVT model probability of exceeding normal training  "
            if(failcnt > 10):
                self.summary += " Uncontrollable dynamics for unknown reasons, but clearly novel as failure frequencey too high compared to training"
            if(not outputstats):                
                self.summary += "#####"

                



        if(self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1458"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            if(self.uccscart.tbdebuglevel>1): print(self.debugstring)
            self.worldchangedacc=self.previous_wc          
                
        if(self.worldchangedacc > self.previous_wc):  self.previous_wc = self.worldchangedacc
        elif(self.worldchangedacc < self.previous_wc):  self.worldchangedacc = self.previous_wc 

        print('Dend# {}  Logstr={} {} Prob={} {} Scores= {}   '.format( self.episode,  self.logstr,
                                                                        "\n", [round(num, 2) for num in self.problist[:40]], 
                                                                        "\n", [round(num, 2) for num in self.scorelist[:40]]))            

        
        if(self.worldchangedacc >.5 and self.uccscart.adapt_after_detect and self.uccscart.adapt_delay >=0):
            self.uccscart.adapt_delay = self.uccscart.adapt_delay -1

        if(self.episode< self.scoreforKL):
            self.worldchangedacc = 0
            self.worldchangeblend = 0
            self.previous_wc = 0                       
            
        return self.worldchangedacc


    def ball_location_error(self, statevector):  # if step 10 or 45, get long term ball position error computed from initial state vs current
        if(self.episode < self.scoreforKL): return 0;  # no need until we start testing for novelty via KL 
        if(not (self.uccscart.tick  ==10 or self.uccscart.tick  ==25 or self.uccscart.tick  ==35 or self.uccscart.tick  ==45 or self.uccscart.tick  ==55 or self.uccscart.tick  ==65 or self.uccscart.tick  ==75)):
            return 0
        err = 0
        zerr=0
        numblocks = int((len(statevector)-13)/6)
        prob=0
        adiffs = np.zeros(3)
        diffs = np.zeros(3)        
        nb=0
        if(self.uccscart.tick == 10):
            self.consecutivehighball=0            
            
        for block in range(numblocks):
            for i in range(3):
                if((i < 2 and abs(statevector[i+block*6+13]) < 4)
                   or (i == 2 and abs(statevector[i+block*6+13]) < 9 and abs(statevector[i+block*6+13]) >1)):
                    diff = self.expected_ball_locaitons[self.uccscart.tick][block][i] - statevector[i+block*6+13]
                    nb += 1
                else: diff =0    #don't consider diffs near boundary as simulation has larger errros as it bounces
                adiffs[i]  += diff*diff
                diffs[i]  += diff
                err  += diff*diff
        #get average error
        if(nb >0):
            err = err / nb
            adiffs = adiffs/nb
            diffs = diffs/nb            
        else:
            adiffs = np.zeros(3)
            diffs = np.zeros(3)        
            err = 0
            self.logstr +=  "&M42 LL8 Block Long-term Location Prediction Error Failure Block Decrease to 0 at  Step " + str(self.tick) +" No blocks cannot compute location error " 
            self.worldchangedacc =  1
            
        zerr = adiffs[2] #last error is just z
        ratio = zerr / err  #get zerror a fraction of total eror

        dirstr = "INCREASE"   # default is Z decreases  which means gravity or direction is an INCREASE
        if(diffs[2] >0): dirstr = "DECREASE"
        prob=0
        zprob=-1
        eprob=-1
        level=0  # setup default
        if((self.uccscart.tick  ==10 or self.uccscart.tick  ==25 or self.uccscart.tick  ==35 or self.uccscart.tick  ==45 or self.uccscart.tick  ==55 or self.uccscart.tick  ==65 or self.uccscart.tick  ==75)):
            if(self.uccscart.tick  ==10):
                zprob = self.rwcdf(zerr,0.23754, 1.123764, 1.2201)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,.46212, 1.0128, .52377)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale
                
                if(prob > .01):             self.consecutivehighball += 1            
                if(prob > .01 and zerr >.32 and ratio > .7 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " + dirstr  + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):
                    level=7
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
            elif (self.uccscart.tick  ==25):  #  gives us much longer to observed deviations.. 
                zprob = self.rwcdf(zerr,2.1132, 1.6435, .85227)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,4.376, 1.4333, .82227)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale
                if(prob > .01):             self.consecutivehighball += 1
                else: self.consecutivehighball =0
                
                if(self.episode > self.scoreforKL and self.episode < self.scoreforKL*3 ):
                    prob = min(self.maxballscaled,prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive
                    #prob = (prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive                     
                    self.problist[2] = self.problist[2] + prob #stick it in problist in location  5 so it can impact KL
                    #self.worldchangedacc += min(prob,.05)  #we give small  direct bump as well
                if(prob >.01 and ratio > .94 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.01 and ratio < .57 ):
                    level=7
                    ##prob += .1   #this ratio does not occur for random errors often                    
                    self.logstr +=  "&M42 LL7 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):            
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " \
                                    + str(round(ratio,2)) +" " + str(round(diffs[0],2)) +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " +  str(self.hint)
            elif (self.uccscart.tick  ==35): 
                zprob = self.rwcdf(zerr,4.03137, 1.4216, .8342)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,10.6324, 1.8217, .6728)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale

                if(prob > .01):             self.consecutivehighball += 1
                else: self.consecutivehighball =0
                if(self.episode > self.scoreforKL and self.episode < self.scoreforKL*3 ):
                    #prob = min(max(prob,self.maxdynamicprob),prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive
                    prob = (prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive                                         
                    self.problist[3] = self.problist[3] + prob #stick it in problist in location  5 so it can impact KL
                    #self.worldchangedacc += min(prob,.05)  #we give small  direct bump as well                    
                if(prob >.01 and ratio > .94 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.01 and ratio < .49 ):
                    level=7
                    ##prob += .1   #this ratio does not occur for random errors often                    
                    self.logstr +=  "&M42 LL7 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):            
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) +" " \
                                    + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " +  str(self.hint)
            elif (self.uccscart.tick  ==45):  #time 45,  gives us much longer to observed deviations.. 
                zprob = self.rwcdf(zerr,5.65032, 1.33756, .8412)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,16.3461, 1.2178, .67227)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale

                if(prob > .01):             self.consecutivehighball += 1
                else: self.consecutivehighball =0 
                if(self.episode > self.scoreforKL and self.episode < self.scoreforKL*2 ):
                    prob = min(self.maxballscaled,prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive 
                    #prob = (prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive                                         
                    self.problist[4] = self.problist[4]+ prob #stick it in problist in location 4  so it can impact KL
                    #self.worldchangedacc += min(prob,.05)  #we give small  direct bump as well                    
                if(prob >.01 and ratio > .95 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.01 and ratio < .49 ):
                    level=7
                    ##prob += .1   #this ratio does not occur for random errors often                    
                    self.logstr +=  "&M42 LL7 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):            
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) +" " \
                                    + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " +  str(self.hint)
            elif (self.uccscart.tick  ==55 and (self.consecutivehighball>0 or err >28)):  #time 55,  gives us much longer to observed deviations.. 
                zprob = self.rwcdf(zerr,8.5218, 1.8256, .9926)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,28.0121, 1.8213, .9627)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale
                
                if(prob > .01):             self.consecutivehighball += 1
                else: self.consecutivehighball =0 
                if(self.episode > self.scoreforKL and self.episode < self.scoreforKL*2 ):
                    prob = min(self.maxballscaled,prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive
                    #prob = (prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive                                         
                    self.problist[5] = self.problist[5]+ prob #stick it in problist in location 4  so it can impact KL
                    #self.worldchangedacc += min(prob,.05)  #we give small  direct bump as well                    
                if(prob >.01 and ratio > .95 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.01 and ratio < .49 ):
                    level=7
                    #prob += .1                    
                    self.logstr +=  "&M42 LL7 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):            
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) +" " \
                                    + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " +  str(self.hint)
            elif (self.uccscart.tick  ==65 and (self.consecutivehighball>0 or err >33.5)):  #time 55,  gives us much longer to observed deviations.. 
                zprob = self.rwcdf(zerr,8.6812, 1.5718, .9421)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,31.512, 1.0133, .87227)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale

                if(prob > .01):             self.consecutivehighball += 1
                else: self.consecutivehighball =0 
                if(self.episode > self.scoreforKL and self.episode < self.scoreforKL*2 ):
                    prob = min(self.maxballscaled,prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive 
                    self.problist[6] = self.problist[6]+ prob #stick it in problist in location 4  so it can impact KL
                    #self.worldchangedacc += min(prob,.05)  #we give small  direct bump as well                    
                if(prob >.01 and ratio > .95 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.01 and ratio < .49 ):
                    level=7
                    #prob += .1   #this ratio does not occur for random errors often                    
                    self.logstr +=  "&M42 LL7 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):            
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) +" " \
                                    + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " +  str(self.hint)
            elif (self.uccscart.tick  ==75 and self.consecutivehighball>0 and zerr < 17):  #time 75,  gives us much longer to observed deviations.. 
                zprob = self.rwcdf(zerr,8.75032, 1.1417, .9772)  #parms by hand  computation based on normal data from prints below
                eprob = self.rwcdf(err,35.7641, 1.0122, .87227)  #parms by hand  computation based on normal data from prints below
                prob = max(zprob,eprob)*self.ballprobscale
                if(prob > .01):             self.consecutivehighball += 1
                else: self.consecutivehighball =0 
                if(self.episode > self.scoreforKL and self.episode < self.scoreforKL*2 ):
                    prob = min(self.maxballscaled,prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive
                    #prob = (prob*(self.consecutivehighball+1)) # we scale up if they have been multiple consecutive                                         
                    self.problist[7] = self.problist[7]+ prob #stick it in problist in location 4  so it can impact KL
                    #self.worldchangedacc += min(prob,.05)  #we give small  direct bump as well                    
                    if(self.uccscart.tbdebuglevel>1 and len(str(self.hint))!=0):
                        print('preStep {}, E {} ball_loc {} p={} {} {} {} R {} hint=|{}| plist={}'.format(self.uccscart.tick,self.episode,round(zerr,2), round(prob,2), round(diffs[0],2),round(diffs[1],2),round(diffs[2],2), ratio,str(self.hint)[9:15],str(self.problist[2:5])))
                if(prob >.01 and ratio > .95 ):
                    level=6
                    self.logstr +=  "&M42 LL6 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.01 and ratio < .49 ):
                    level=7
                    #prob += .1   #this ratio does not occur for random errors often                    
                    self.logstr +=  "&M42 LL7 Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) \
                                    +" " + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " 
                elif(prob >.02):            
                    self.logstr +=  "&M42 LL6 LL7 (unclear) Block Long-term Location Prediction Error " +dirstr + " Step " + str(self.tick) +" " \
                                    + str(round(zerr,2)) +" p" + str(round(prob,2)) +" " + str(round(ratio,2)) +" " + str(round(diffs[0],2)) +" " \
                                    + str(round(diffs[1],2)) +" " + str(round(diffs[2],2)) + " " +  str(self.hint)
            elif (self.uccscart.tbdebuglevel>1 and len(str(self.hint))!=0):
                print('Unhandled Step {}, E {} ball_loc {} p={} {} d={} {} {} con {} R {} hint=|{}| plist={}'.format(self.uccscart.tick,self.episode,round(zerr,2), round(zprob,2),round(eprob,2), round(diffs[0],2),round(diffs[1],2),round(diffs[2],2),self.consecutivehighball, ratio,str(self.hint)[9:15],str(self.problist[2:8])))                            
                sys.stdout.flush()                


                        
        if(self.uccscart.tbdebuglevel>1 and len(str(self.hint))!=0):
            print('Step {}, E {} ball_loc {} p={} {} d={} {} {} con {} R {} hint=|{}| plist={}'.format(self.uccscart.tick,self.episode,round(zerr,2), round(zprob,2),round(eprob,2), round(diffs[0],2),round(diffs[1],2),round(diffs[2],2),self.consecutivehighball, ratio,str(self.hint)[9:15],str(self.problist[2:8])))                            
            sys.stdout.flush()
        return prob
               
    

    
    def process_instance(self, oactual_state):
        #        pertub = (self.cnt > 100) and (self.maxprob < .5)
        pertub = False
        current = self.current_state=oactual_state # mostly for debugging
        #self.statelist[self.cnt] = oactual_state  # save all states, used in testing to see if trajectory is dynamic or balistic,. also useful for debugging u
        self.probvector= np.zeros(500)
        self.nextprob=15
            
        if(self.saveframes):        
            image= oactual_state['image']
        self.logstr += self.uccscart.char  #copy overy any information about collisions
#        if(len(self.uccscart.char)>0): print("Process inst with char", self.uccscart.char)
        probability = self.uccscart.wcprob  # if cart control detected something we start from that estiamte 
        if(self.uccscart.wcprob ==1):
            self.worldchangedacc=1
            self.worldchanged=1            
            self.worldchangeblend=1

#        if("CP in        self.uccscart.char):
#            self.uccscart.lastscore = 0.001111; #  if we had a lot fo collision potential, ignore the score. 
        self.scorelist.append(self.uccscart.lastscore)        
        self.uccscart.char = ""  #reset any information about collisions
        action, expected_state = self.takeOneStep(oactual_state,
                                                  self.uccscart,
                                                  pertub)

        # we can now fill in previous history's actual 
        if(self.uccscart.force_action >=0 and self.uccscart.force_action < 5):
            self.uccscart.action_history[self.uccscart.force_action][1] = self.uccscart.format_data(oactual_state)        

        #we don't reset in first few steps because random start may be a bad position yielding large score
        #might be were we search for better world parmaters if we get time for that
        #TB.. this does not seem to be needed as reset hasppend when searching for best action 
#        if(self.cnt > self.skipfirstNscores and self.uccscart.lastscore > self.scoretoreset):
#            print("At step ", self.cnt, "resettin to actual because of a large score", self.uccscart.lastscore)
#            self.uccscart.reset(actual_state)



        raw_data_val = self.prev_predict
        self.prev_predict = expected_state
        self.prev_state = oactual_state        
        self.cnt += 1
        if (self.cnt == 1):  # if first run cannot check dynamics just initial state
            if(self.uccscart.tbdebuglevel>0):            
                self.debugstring = 'Testing initial state for obvious world changes: actual_state={}, next={}, dataval={}, '.format(oactual_state,
                                                                                                                                    expected_state,
                                                                                                                                    raw_data_val)
            self.expected_ball_locaitons=self.uccscart.get_expected_ball_states(oactual_state)


            initprob= self.istate_diff_EVT_prob(oactual_state)
            
            #update max and add if initprob >0 add list (if =0 itnore as these are very onesided tests and don't want to bias scores in list)
            self.maxprob = max(initprob, self.maxprob)

            if(initprob >0):
                self.problist.append(initprob)  # add a very big bump in prob space so KL will see it                
                #if worldis known to have changed and initprob see something, then start with avoid
                #if(self.worldchangedacc>.5):
                #    self.use_avoid_reaction=True
                    
                if(self.uccscart.tbdebuglevel>0):
                    print('Init probability checks set prob to 1 with actual_state={}, next={}, dataval={}, problist={}, '.format(oactual_state,
                                                                                                                                expected_state,
                                                                                                                                raw_data_val,
                                                                                                                                self.problist))

                # if (self.debug):
                #     self.debugstring = 'Early Instance: actual_state={}, next={}, dataval={}, '.format(oactual_state,expected_state,raw_data_val)
                self.prev_action = action
                return action
        else:

            data_val = self.format_data(raw_data_val)
            prob_values = []
            actual_state = self.format_data(oactual_state)

            #only look for ball changes early,  else save the expense
            bprob = self.ball_location_error(actual_state)
            probability += bprob  #also include it in the probability for KL-testing
            
            
            #if sizes changed then we have different number of blocks.. and it must be novel
            if(len(data_val) != len(actual_state)):
                probability = 1.0
                self.uccscart.characterization['entity']="Block"; 
                self.uccscart.characterization['attribute']="quantity";
                if(len(data_val) >= len(actual_state)):
                    if(self.logstr.count("LL8") >1 ):  #if we had more than one chance its increasing
                        self.uccscart.characterization['change']='decreasing';
                    else:
                        self.uccscart.characterization['change']='decrease';                        
                else:
                    if(self.logstr.count("LL8") >1 ):
                        self.uccscart.characterization['change']='increase';
                    else:
                        self.uccscart.characterization['change']='increasing';                
                self.worldchangedacc=1
                tstring= " & M42 LL8: Blocks quantity " + str(self.uccscart.characterization['change']) + " FV len "+ str(len(data_val)) + " changed to " + str(len(actual_state))
                if(self.logstr.count("LL8") < 2): self.logstr += str(tstring)
                if(self.uccscart.tbdebuglevel>1): print(tstring)
                
                
            else:
                # vectors can be subtracted
                difference_from_expected = data_val - actual_state  # next 4 are the difference between expected and actual state after one step, i.e.
                current = difference_from_expected

                
                diffprobability = self.dynam_prob_scale * self.cstate_diff_EVT_prob(current,actual_state)
                probability += self.dynblocksprob  # blocks are less noisy so we always add them in

                if(self.uccscart.tbdebuglevel>1 and probability>.05):
                    print("E/S " + str(self.episode)+"."+str(self.tick) +" ball dyncnt diff porb and overall prob ", bprob,self.dynamiccount, diffprobability, probability)                                    
                self.probvector[self.nextprob] = probability
                self.nextprob += 1                    
                
                
                # if dynamics is really off, it should be off almost every time, so only use it when we have a large count and limit its growth, let KL detect it
                if(self.dynamiccount > 4 and diffprobability>0 ):
                    probability = min(self.maxdynamicprob,probability+ diffprobability)
                    self.dynamiccount = self.dynamiccount+1   # if we don't see dyamics reduce count                    
                    if(self.uccscart.tbdebuglevel>0):
                        print(" In step " + str(self.tick) +" usin dyn prob ", self.dynamiccount, diffprobability, probability)                    
                else:
                    if(diffprobability>.05 and self.uccscart.tbdebuglevel>0 ):
                        print(" Step " + str(self.tick) +" skipping low dyncnt and its prob ", self.dynamiccount, diffprobability, probability)
                        self.dynamiccount = self.dynamiccount+1   # if we don't see dyamics reduce count                    
                if(diffprobability<=0.05 and self.dynamiccount>2):                        
                    self.dynamiccount = self.dynamiccount-1   # if we don't see dyamics reduce count

                self.probvector[self.nextprob] = probability
                self.nextprob += 1                    



                        

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
            #         if(actprob>0): self.logstr += "Action search with prob " + str(actprob)


#####!!!!!#####  end GLUE/API code EVT-
#####!!!!!#####  Start domain dependent adaption




            # if we have not had a lot successess in a row (sign index is right)   and declared world changed and  we ahve enough failures then try another index
            if((not (self.uccscart.never_adapt or (self.uccscart.adapt_delay >0))) and
                self.maxconsecutivesuccess < 2 and  self.maxconsecutivefail > self.maxconsecutivefailthresh and  self.consecutivefail > 1 ):
               # try the next permuation.. see if we can reduce the fail rate
               self.uccscart.actions_permutation_index += 1
               if(self.uccscart.actions_permutation_index > (len(self.uccscart.actions_plist)-1)):
                   self.uccscart.actions_permutation_index = 0                    
               self.logstr += "#####? Too many failures.  Guessing actions were mapped/perturbed.. Now using pertubation " 
               self.logstr.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
               self.logstr += "if this is the last time you see this message and performance is now good then characterize this as the action permutation in placeof the  uncontrollable characateration provided after world change provided earlier #####?"
               print(self.logstr)
               self.consecutivefail = 0
            

            #if we have a permutation index and we lasted more then 40 time steps, we are succesful in control and the world probably changed. 
            if(self.uccscart.actions_permutation_index> 0 and self.cnt >40):
                probability = 1
                self.worldchangedacc += .01;



            probability = min(1,probability)
            self.problist.append(probability)


            self.maxprob = max(probability, self.maxprob)

            self.probvector[self.nextprob] = self.maxprob
            self.nextprob += 1                    
            

            # we can also include the score from control algorithm,   we'll have to test to see if it helps..
            #first testing suggests is not great as when block interfer it raises score as we try to fix it but then it seems novel. 
            #                self.maxprob=min(1,self.maxprob +  self.uccscart.lastscore / self.scalelargescores)
            if (self.cnt > 0 and len(self.problist)>0 ):
                self.meanprob = np.mean(self.problist)

            if (self.uccscart.tbdebuglevel>3):
                self.debugstring = 'Instance: cnt={},actual_state={}, next={},  current/diff={},NovelProb={}'.format(
                    self.cnt, actual_state, expected_state, current, probability)
                print("Step prob/probval",self.tick,probability, prob_values, "maxprob", self.maxprob, "meanprob", self.meanprob)  
            
        self.prev_action = action

        if(self.saveframes):        
#            image = feature_vector['image']  #done at begining now
            if image is None:
#                self.log.error('No image received. Did you set use_image to True in TA1.config '
#                               'for cartpole?')
                print('No image received. Did you set use_image to True in TA1.config '
                      'for cartpole?')
                sys.stdout.flush()                                
                found_error = True
                
            else:
                s = 720.0 / image.shape[1]
                endy=int(image.shape[0] * s)
                dim = (720, endy )
                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (10, 30)
                fontScale = .5
                # Blue color in BGR
                if(round(self.worldchangedacc,3) < .5):            color = (255, 0, 0)
                else  :            color = (0,0,255)
                thickness = 2
                fname = '/scratch/tboult/PNG/{1}-Frame-{0:05d}.png'.format(self.framecnt,self.saveprefix)            
                wstring = 'E={4:03d}.{0:03d} CP={7:4.3f}  WC={2:4.3f} P={1:3.2f} N={6:.1} I={9:01d} A={8:02d}, S={3:12.6} L={5:.8},'.format(self.uccscart.tick,probability,self.worldchangedacc,self.uccscart.lastscore,self.episode,self.logstr[-8:], str(self.noveltyindicator),100*self.perf/(max(1,self.totalcnt)),self.uccscart.adapt_level,self.uccscart.actions_permutation_index        )            
                outimage = cv2.putText(resized, wstring, org, font,
                                       fontScale, color, thickness, cv2.LINE_AA)

                #some constants so we know there is something in array
                self.probvector[self.nextprob] = self.worldchangedacc; self.nextprob += 1
                self.probvector[self.nextprob]= self.uccscart.adapt_level/4.0; self.nextprob += 1
                self.probvector[self.nextprob] = probability; self.nextprob += 1
                self.probvector[self.nextprob] = min(1,self.uccscart.lastscore/100); self.nextprob += 1                
#                print("lastprob at ", self.nextprob)

                self.nextprob=0            
                self.probvector[self.nextprob] = self.trialchar.count("init")/200; self.nextprob += 1
                self.probvector[self.nextprob]= self.trialchar.count("Block")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("Block Vel")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("Block Motion")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("Pole")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("Cart")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("small")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("INCREASE")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("DECREASE")/200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("large") / 200; self.nextprob += 1                                
                self.probvector[self.nextprob] = self.trialchar.count("diff") / 200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("Vel") / 200; self.nextprob += 1                                
                self.probvector[self.nextprob] = self.trialchar.count("High") / 200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("attacking") / 200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("aiming") / 200; self.nextprob += 1
                self.probvector[self.nextprob] = self.trialchar.count("parallel")/ 200; self.nextprob += 1


                
#                cmap = matplotlib.cm.get_cmap('jet')
                cmap = matplotlib.cm.get_cmap('nipy_spectral')                

                for i in range(60):
                    dcolor = cmap(self.probvector[i])
#                    print("i prob color", i, self.probvector[i], dcolor)
                    cv2.rectangle(outimage,(i*12,endy-12),((i+1)*12,endy),(255*dcolor[0],255*dcolor[1],255*dcolor[2]),-1)

                cv2.imwrite(fname, outimage)
                


                self.framecnt += 1
                if ((self.uccscart.tbdebuglevel>-1 )and self.framecnt < 3):
                    self.debugstring += '  Writing '+ fname + 'with overlay'+ wstring
                    print(self.debugstring)
                #        if (self.uccscart.tbdebuglevel>1) :
                #            print(' Episode (), step{} final probvector ={}'.format( self.episode, [round(num, 2) for num in self.probvector]))
                
                    
            
        sys.stdout.flush()                
        
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
#                 diffprobability[action][index] = self.dynam_prob_scale * self.cstate_diff_prob(statediff[action][index])


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

#         #if there is a pertubation with lower overall error than        
#         if(minprob < probs[self.uccscart.actions_permutation_index]):
#             self.uccscart.actions_permutation_index = index   #keep that index
#             self.logstr += "Actions were perturbed.. Now using pertubation " 
#             self.logstr.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
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
#             diffprobability = self.dynam_prob_scale * self.cstate_diff_prob(statediff,actual_state)
#             if(diffprobability < bestprob):            
#                 if(diffprobability < .0005): # stop early if good score
#                     self.logstr += "Actions were perturbed.. Now using pertubation " 
#                     self.logstr.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
#                     print("Good pertubation ", self.uccscart.actions_plist[self.uccscart.actions_permutation_index],"with index/tried", self.uccscart.actions_permutation_index, self.uccscart.actions_permutation_tried )                    
#                     return 1;
#                 else:
#                     bestprob = diffprobability
#                     bestindex = self.uccscart.actions_permutation_index
           

#         self.uccscart.reset(actual_state)                    #back to normal state

#         if(bestprob  < diffprobability and bestprob < .01): # not great but  better than where we started and maybe good enough. 
#             self.uccscart.actions_permutation_index = bestindex
#             print("Now using pertubation ", self.uccscart.actions_plist[self.uccscart.actions_permutation_index],"with index/tried", self.uccscart.actions_permutation_index, self.uccscart.actions_permutation_tried )
#             self.logstr += "Actions might be  perturbed.. Now using pertubation "
#             self.logstr.join(map(str,self.uccscart.actions_plist[self.uccscart.actions_permutation_index]))
#             self.logstr += "with prob" + str(bestprob)
#             return bestprob;

#         #else nothing work so reset back to initial action list
#         self.uccscart.actions_permutation_index = 0
#         self.uccscart.actions_permutation_tried = np.zeros(len(self.uccscart.actions_permutation_tried))
        
#         return 0



