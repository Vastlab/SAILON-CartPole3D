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
        self.scoreforKl=10        # we only use the first sets of scores for KL because novels worlds close when balanced
        self.num_epochs = 200
        self.num_dims = 4
        self.scalelargescores=20
        # takes a while for some randome starts to stabilise so don't reset too early as it
        # reduces world change sensitvity        
        self.skipfirstNscores=2
        self.worldaccscale = .5        

        self.skipfail=4  #no penalty for up to this many failures,   larger is more robsut for non-novel worlds so should be close to its expected failure rate.  start at 200 to see raw failure rate than set it based on that
        self.failscale=5 #   we scale failures like (self.failcnt-self.skipfail)/self.failscale  and add as absolute offset ot blened estimate or world change.
        self.initfailscale=.2 #   we scale prob from initial state by this amount and add world accumulator each time. With .05 even a prob of 1 will take 20 steps to detect..  but this balances risk from going of on non-novel worlds


        # Large "control scores" often mean things are off, since we never know the exact model we reset when scores get
        # too large in hopes of  better ccotrol
        self.scoretoreset=20
        


        # TODO: change evm data dimensions
        if (self.num_dims == 4):
            self.mean_train = 0
            self.stdev_train = 0.0
            self.prob_scale = 2  # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
        else:
            self.mean_train = .198   #these are old values from Phase 1 2D cartpole..  for Pahse 2 3D we compute frm a training run.
            self.stdev_train = 0.051058052318592555
            self.mean_train = .001   #these guessted values for Phase 2 incase we get called without training
            self.stdev_train = 0.000000001
            self.prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.

        self.cnt = 0
        self.worldchanged = 0
        self.worldchangedacc = 0
        self.failcnt = 0        
        self.worldchangeblend = 0
        # from WSU "train".. might need ot make this computed.
        #        self.mean_train=  0.10057711735799268
        #       self.stdev_train = 0.00016
        self.problist = []
        self.scorelist=[]
        self.maxprob = 0
        self.meanprob = 0
        self.expected_backone = np.zeros(4)
        self.noveltyindicator = None
        self.correctcnt=0
        self.rcorrectcnt=0        
        self.totalcnt=0        
        self.perf=0                
        #        self.expected_backtwo = np.zeros(4)
        self.episode = 0
        self.trial = 0
        self.given = False
        self.statelist = []
        self.debug = True
        self.debugstring = ""
        self.character=""
        
        # Create prediction environment
        env_location = importlib.util.spec_from_file_location('CartPoleBulletEnv', \
                                                              'cartpolepp/new2dcart.py')
        env_class = importlib.util.module_from_spec(env_location)
        env_location.loader.exec_module(env_class)
        self.env_prediction = env_class.CartPoleBulletEnv()
        self.env_prediction.path = "./cartpolepp"
        with open('evm_config.json', 'r') as json_file:
            evm_config = json.loads(json_file.read())
            cover_threshold = evm_config['cover_threshold']
            distance_multiplier = evm_config['distance_multiplier']
            tail_size = evm_config['tail_size']
            distance_metric = evm_config['distance_metric']
            torch.backends.cudnn.benchmark = True
            args_evm = argparse.Namespace()
            args_evm.cover_threshold = [cover_threshold]
            args_evm.distance_multiplier = [distance_multiplier]
            args_evm.tailsize = [tail_size]
            args_evm.distance_metric = distance_metric
            args_evm.chunk_size = 200

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
        self.env_prediction.resetbase()
        self.env_prediction.reset()
        
        if(episode ==0):  #reset things that we carry over between episodes withing the same trial
            self.worldchangedacc = 0
            self.failcnt = 0                    
            self.worldchangeblend = 0            
            self.worldaccscale = .5                    


        # Take one step look ahead, return predicted environment and step

    # env should be a CartPoleBulletEnv
    def takeOneStep(self, state_given, env, pertub=False):
        observation = state_given
        action, next_action, expected_state = env.get_best_action(observation,self.meanprob)
        # if doing well pertub it so we can better chance of detecting novelties
        '''ra = int(state_given[0] * 10000) % 4
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

    def format_data(self, feature_vector):
        # Format data for use with evm
        state = []
        for i in feature_vector.keys():
            if i != 'blocks' and i != 'time_stamp' and i != 'image':
                #print(feature_vector[i])
                for j in feature_vector[i]:
                    state.append(feature_vector[i][j])
                #print(state)
        return np.asarray(state)

    # get probability differene froom initial state
    def istate_diff_prob(self,actual_state):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart x'" , " Cart y'" , " Cart x'" ,  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole x'" , " Pole y'" , " Pole z'" , " Block x" , " Block y" , " Block x" ,  " Block x'" , " Block y'" , " Block x'" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]
        
        #load imin/imax from training..  with some extensions. From code some of these values don't seem plausable (blockx for example) but we saw them in training data.  maybe nic mixed up some parms/files but won't hurt too much fi we mis some
        imin =  np.array([-2.991261, -2.997435,  0.1     , -0.491182, -0.019649,  0.      ,
                            -0.010097, -0.011602, -0.014616,  -0.999822, -0.032205, -0.177146,
                            -0.530645, -4.14159 , -4.148426,  -0.837303, -20, -20,      
                            -20, -5.      , -5.      ,  0.      ,  5.      , -5.      ,
                            0.      ,  5.      ,  5.      ,  0.      , -5.      ,  5.      ,
                            0.      , -5.      , -5.      , 10.      ,  5.      , -5.      ,
                            10.      ,  5.      ,  5.      , 10.      , -5.      ,  5.      ,
                            10.      ])
        imax =  np.array([ 3.000305e+00,  2.999666e+00,  1.000000e-01,  5.306980e-01,
                              1.966100e-02,  0.000000e+00,  1.012700e-02,  8.566000e-02,
                              1.398700e-02,  9.999990e-01,  3.111000e-02, 1.205450e-01,
                              4.910830e-01,  4.143999e+00,  4.157679e+00,  9.968307e+00,
                              20,  20,  20, -5.000000e+00,
                              -5.000000e+00,  0.000000e+00,  5.000000e+00, -5.000000e+00,
                              0.000000e+00,  5.000000e+00,  5.000000e+00,  0.000000e+00,
                              -5.000000e+00,  5.000000e+00,  0.000000e+00, -5.000000e+00,
                              -5.000000e+00,  1.000000e+01,  5.000000e+00, -5.000000e+00,
                              1.000000e+01,  5.000000e+00,  5.000000e+00,  1.000000e+01,
                              -5.000000e+00,  5.000000e+00,  1.000000e+01])
        #note we mult by 20% to allow slight variations from training.
        imax = imax + .1*abs(imax)
        imin = imin - .1*abs(imin)        
        
        
        initprob=0  # assume nothing new in world
        #check if any abs (block velocities) < 6 > 10 in any dimension   Done as a hack.. better to do EVT fitting on  values based on training ranges. 
        cart_pos = [actual_state['cart']['x_position'],actual_state['cart']['y_position'],actual_state['cart']['z_position']]
        cart_pos = np.asarray(cart_pos)

        
        istate = self.format_istate_data(actual_state)
        # do base state for cart(6)  and pole (7) 
        for j in range (13):
            if(istate[j] > imax[j]):
                probv=  (istate[j] - imax[j]) / (abs(istate[j]) + abs(imax[j]))
                if(probv>1e-6):
                    initprob += probv
                    self.character += str(dimname[j]) + " inital too large  "+ " " + str(round(istate[j],3)) +" " + str(round(imax[j],3)) +" " + str(round(probv,3))
  
                
            if(istate[j] < imin[j]):
                probv=  (imin[j] - istate[j])/ (abs(istate[j]) + abs(imin[j]))
                if(probv>1e-6):
                    initprob += probv
                    self.character += str(dimname[j]) + " inital too small  "+ " " + str(round(istate[j],3)) +" " + str(round(imin[j],3))+" " + str(round(probv,3)) 
        
        wallstart= len(istate) - 24 
#        pdb.set_trace()
        k=19 # for name max/ame indixing where we have only one block
        for j in range (wallstart,len(istate),1):
            if(istate[j] > imax[k]):
                probv=  (istate[j] - imax[k])/ (abs(istate[j]) + abs(imax[k]))
                if(probv>1e-6):
                    initprob += probv
                    self.character += str(dimname[k]) + " inital too large  " + " " + str(round(istate[j],3)) +" " + str(round(imax[k],3))+" " + str(round(probv,3))
            if(istate[j] < imin[k]):
                probv=  (imin[k] - istate[j])/ (abs(istate[j]) + abs(imin[k]))
                if(probv>1e-6):
                    initprob += probv
                self.character += str(dimname[k]) + " inital too small  " + " " + str(round(istate[j],3)) +" " + str(round(imin[k],3))+" " + str(round(probv,3))
            k = k +1

                    
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,wallstart,1):
            if(istate[j] > imax[k]):
                probv =  (istate[j] - imax[k]) / (abs(istate[j]) + abs(imax[k]))                
                if(probv>1e-6):
                    initprob += probv
                    self.character += str(dimname[k]) + " inital too large " +  " " + str(round(istate[j],3)) +" " + str(round(imax[k],3)) +" " + str(round(probv,3))
            if(istate[j] < imin[k]):
                probv=  (imin[k] - istate[j])/ (abs(istate[j]) + abs(imin[k]))
                if(probv>1e-6):
                    initprob += probv
                    self.character += " " + str(dimname[k]) + " inital too small " +  " " + str(round(istate[j],3)) +" " + str(round(imin[k],3)) +" " + str(round(probv,3))
            k = k +1
            if(k==19): k=13;   #reset for next block
        return initprob


    def wcdf(self,x,imean,ishape,iscale):
        prob = 1-math.pow(math.exp(-(x-imean)/iscale),ishape)
#        if(prob > 1e-4): print("in wcdf",round(prob,6),x,imean,ishape,iscale)
        return prob
    


    # get probability differene froom continuing state difference
    def cstate_diff_prob(self,cdiff):
        dimname=[" Cart x" , " Cart y" , " Cart z" ,  " Cart x'" , " Cart y'" , " Cart x'" ,  " Pole x" , " pole y" , " Pole z" ," Pole w" ,  " Pole x'" , " Pole y'" , " Pole z'" , " Block x" , " Block y" , " Block x" ,  " Block x'" , " Block y'" , " Block x'" , " Wall 1x" ," Wall 1y" ," Wall 1z" , " Wall 2x" ," Wall 2y" ," Wall 2z" , " Wall 3x" ," Wall 3y" ," Wall 3z" , " Wall 4x" ," Wall 4y" ," Wall 4z" , " Wall 5x" ," Wall 5y" ," Wall 5z" , " Wall 6x" ," Wall 6y" ," Wall 6z" , " Wall 8x" ," Wall 8y" ," Wall 8z" , " Wall 9x" ," Wall 9y" ," Wall 9z" ]
        
        #load imin/imax from training.. 
        # imin = np.array([-1.00000000e-05, -6.25000000e-03,  -1.00000000e-06 -1.00000000e-05,
        #                     -9.05000000e-03,  -1.00000000e-06, -7.00000000e-05, -7.00000000e-05,
        #                     -1.00000000e-05,  -1.00000000e-06, -6.64000000e-03, -6.69000000e-03,
        #                     -1.00000000e-05, -1.00000000e-05, -3.33333333e-06, -6.66666667e-06,
        #                     -6.66666667e-06, -6.66666667e-06, -6.66666667e-06])
        imax =   1.20* np.array([1.0000e-04, 4.0100e-03, 1.0000e-06, 1.0000e-04, 3.6450e-03,
                             1.0000e-06, 1.2000e-04, 2.4600e-03, 1.0000e-06, 2.0000e-03,
                             2.1590e-03, 2.4606e-01, 1.0000e-06, 5.0000e-06, 5.0000e-06,
                             5.0000e-06, 5.0000e-06, 5.0000e-06, 7.5000e-06])
        imin = -imax

        #TB hand computed from nic's data.  not enough data but a start with weibull, 

        iscale = np.array([3.47155906e-01, 3.31091666e-01, 1.00000000e-01, 1.73532410e-01,
                      7.52148062e-01, 1.00000000e-01, 4.59669985e-01, 5.41859264e-01,
                      1.60399038e-01, 2.28125787e-01, 4.72931708e-01, 5.40637291e-01,
                      1.60559395e-01, 1.41297786e-01, 4.28488515e-01, 1.62137050e-01,
                           7.06490146e-01, 2.14242695e-01, 8.10682558e-01])

        ishape =      np.array([5.91607978e-01, 2.57201447e-01, 1.00000000e+01, 5.23808171e-01,
                                8.58521831e-02, 1.00000000e+01, 3.49034382e-01, 5.20221998e-01,
                                5.38516481e-01, 4.94342998e-01, 6.74897395e-01, 5.21548155e-02,
                                4.97493719e-01, 3.28189478e-01, 2.52625107e-01, 2.68224616e-01,
                                2.74367961e-01, 3.21022671e-01, 2.78388218e-01])

        
        
        prob=0  # assume nothing new in world
        #check if any abs (block velocities) < 6 > 10 in any dimension   Done as a hack.. better to do EVT fitting on  values based on training ranges. 

        istate = cdiff
        # do base state for cart(6)  and pole (7) 
        for j in range (13):        
            if(istate[j] > imax[j]):
                probv =  self.wcdf(istate[j],imax[j],iscale[j],ishape[j]);
                if(probv>1e-6):
                    self.character += dimname[j] + " above max change, prob " + " " + str(round(probv,3)) + " s/l " + str(round(istate[j],3)) +  " " + str(round(imax[j],3))
                prob += probv
            elif(istate[j] < imin[j]):
                probv =  self.wcdf(-istate[j],-imin[j],iscale[j],ishape[j]);
                if(probv>1e-6):
                    self.character += dimname[j] + " below min change, prob " + " " + str(round(probv,3)) + "  s/l " + str(round(istate[j],3)) +  " " + str(round(imin[j],3))
                prob += probv
        

        #no walls in dtate diff just looping over blocks
        
        k=13 # for name max/ame indixing where we have only one block
        for j in range (13,len(istate),1):
            if(istate[j] > imax[k]):
                probv =  self.wcdf(istate[j],imax[k],iscale[j],ishape[j]);
                if(probv>1e-6):
                    self.character += " " + str(dimname[k]) + " above max change, prob " + " " + str(round(probv,3)) + "  s/l " + str(round(istate[j],3)) +  " " + str(round(imax[j],3))
                prob += probv
            elif(istate[j] < imin[k]):
                probv =  self.wcdf(-istate[j],-imin[k],iscale[j],ishape[j]);
                if(probv>1e-6):
                    self.character += " " + str(dimname[k]) + " below min change, prob " + " " + str(round(probv,3)) + " s/l " + str(round(istate[j],3)) +  " " + str(round(imin[j],3))
                prob += probv
                k = k +1
                if(k==19): k=13;   #reset for next block
#        if(prob > 1e-4): print("Prob", prob)
        return prob
    
                    


    def world_change_prob(self,settrain=False):
        mlength = len(self.problist)
        mlength = min(self.scoreforKl,mlength)
        # we look at the larger of the begging or end of list.. world changes most obvious at the ends. 
        

        if(mlength >0) :
            mu = np.mean(self.problist[:mlength])
            sigma = np.std(self.problist[:mlength])
        else:
            mu = sigma = 0
            self.debugstring = '   ***Zero Lenth World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
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
           self.mean_train = 0.006   #these guessted values for Phase 2 incase we get called without training
           self.stdev_train = 0.028
           self.prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.


#        if (len(self.problist) < 3):
#            print("Very short, world must have changed")
#            return 1;
        if (len(self.problist) < 198):   #for real work
            if (self.debug):
                self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)                
                self.debugstring = '   ***Short World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}        scores{}'.format(
                     round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
                print(self.debugstring)
        if (sigma == 0):
            if (mu == self.mean_train):
                self.debugstring = '      World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}         scores{}'.format(
                     round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
                print(self.debugstring)                
                return 0;
            else:
                sigma = self.stdev_train

            #        pdb.set_trace()
        if(mu < self.mean_train):   #no point computing if world differences are smaller, it may be "much" smaller but that is okay
            self.KL_val = 0   
        else: 
            self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)
        KLscale = (
                              self.num_epochs + 1 - self.episode / 2) / self.num_epochs  # decrease scale (increase sensitvity)  from start down to  1/2
        prob = min(1.0, KLscale * self.KL_val / (2 * self.KL_threshold))
        if (self.debug):
                self.debugstring = '      World Change Acc={}, Prob ={},,mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}         scores{}'.format(
                     round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n",[round(num,2) for num in self.scorelist])
                print(self.debugstring)                
        
        #        self.worldchanged = max(prob,self.worldchanged)

        #for very short runs we scale probablilty because  we did not have enough data for a good KL test.  Inc  accumulator scaling so  we count it more if it keeps happening
        # if (len(self.problist) < self.scoreforKl):
        #     self.worldchanged = prob * len(self.problist)/self.scoreforKl        
#for very short runs we scale probablilty because  we did not have enough data for a good KL test.  Inc  accumulator scaling so  we count it more if it keeps happening        

            
        if (len(self.problist) < self.scoreforKl):
            self.worldchanged = prob * len(self.problist)/(self.scoreforKl)
            self.worldaccscale =  .5+(self.scoreforKl-len(self.problist))/self.scoreforKl
        else:
            self.worldchanged = prob
            self.worldaccscale =  .5            

            
            

        #world change blend can go up or down depending on how probablites vary.. goes does allows us to ignore spikes from uncommon events. as the bump i tup but eventually go down. 
        self.worldchangeblend = min(1, (.25 *self.worldchanged + +.25 * self.worldchangedacc + .5*self.worldaccscale * self.worldchangeblend)/(1 + self.worldaccscale))
        #final result is monotonicly increasing, and we add in an impusle each step if the first step had initial world change.. so that accumulates over time
        if(        len(self.problist) > 0 ) :
            self.worldchangedacc = min(1,self.problist[0]*self.initfailscale + max(self.worldchangedacc,self.worldchangeblend+max(0, (self.failcnt-self.skipfail)/self.failscale )))
        else:             self.worldchangedacc = min(1,max(self.worldchangedacc,self.worldchangeblend+max(0, (self.failcnt-self.skipfail)/self.failscale )))
        return self.worldchangedacc

    def process_instance(self, actual_state):
        #        pertub = (self.cnt > 100) and (self.maxprob < .5)
        pertub = False
        forcedreset=False
#        pdb.set_trace()
        if(self.cnt ==0 ):
            observation = self.env_prediction.reset(actual_state)    #TB  its more sensitive to pertubations if we don't reset after first step

        self.character += self.env_prediction.char  #copy overy any information about collisions
        if("!!!!" in        self.env_prediction.char):
            self.env_prediction.lastscore = 0.001111; #  if we had a lot fo collision potential, ignore the score. 
        self.scorelist.append(self.env_prediction.lastscore)        
        self.env_prediction.char = ""  #reset any information about collisions
        action, expected_state = self.takeOneStep(actual_state, self.env_prediction, pertub)


        #we don't reset in first few steps because random start may be a bad position yielding large score
        #might be were we search for better world parmaters if we get time for that
        if(self.cnt > self.skipfirstNscores and self.env_prediction.lastscore > self.scoretoreset):
#            print("At step ", self.cnt, "resettin to actual because of a large score", self.env_prediction.lastscore)
            self.env_prediction.reset(actual_state)
            forcedreset=True            


        data_val = self.expected_backone
        self.expected_backone = expected_state
        self.cnt += 1
        if (self.cnt < 3):  # skip prob estiamtes for the first ones as we need history to get prediction
            if (self.cnt == 1):
                self.debugstring = 'Testing initial state for obvious world changes: actual_state={}, next={}, dataval={}, '.format(actual_state,
                                                                                                                                    expected_state,
                                                                                                                                    data_val)
                initprob= self.istate_diff_prob(actual_state)

                # for block in actual_state["blocks"]:
                #     pos = [block["x_position"],
                #            block["y_position"],
                #            block["z_position"]]
                #     vel = [block["x_velocity"],
                #            block["y_velocity"],
                #            block["z_velocity"]]
                #     pos = np.asarray(pos)                     
                #     # in normal world  in z block were  (+-4)(+5) so in range 1 to 9 and
                #     # in normal world  in cart never clsoer than 1 unit
                #     #gravity impact things and they did one step before sending us stuff so leave a little room in test from ideal conditions and slowly grow initprob so 1 random thing is nt enough                    
                #     if(np.linalg.norm(cart_pos[0:2] - pos[0:2]) < .9):initprob += 1 - (np.linalg.norm(cart_pos[0:2] - pos[0:2])) ; self.bcharacter += "block norm too close;"; #print('norm fail',initprob,block)
                #     if(pos[2] < .9): initprob += 1 - pos[2] ;  self.character += "block pos2 too smalll;";  #print('pos2 fail',initprob,block)
                #     if(pos[2] > 10.1): initprob +=  pos[2]-10; self.character += "block pos2 too large;"; #print('pos2 fail',initprob,block)  
                #     if(abs(pos[0])>4.1): initprob +=  abs(4 - abs(pos[0]));self.character += "block pos0 out of range;";#print('pos0 fail',initprob,block)
                #     if(abs(pos[1])>4.1): initprob +=  abs(4 - abs(pos[1]));self.character += "block pos1 out of range;";#print('pos1 fail',initprob,block)                                                        


                #     #gravity impact velocities faster allow greater error 
                #     if(abs(abs(vel[0]) - 7.5 ) > 3): initprob += abs(abs(vel[0]) - 7.5 );self.character += "block vel0 out of range;"; #print('vel0 fail',initprob,block)
                #     if(abs(abs(vel[1]) - 7.5 ) > 3): initprob += abs(abs(vel[1]) - 7.5 ); self.character += "block vel1 out of range;";#print('vel2 fail',initprob,block)
                #     if(abs(abs(vel[2]) - 7.5 ) > 3): initprob += abs(abs(vel[2]) - 7.5 ); self.character += "block vel2 out of range;";#print('vel2 fail',initprob,block)                    

                # #any other inital checks go here.. updating  initprob

                #update max and add if initprob >0 add list (if =0 itnore as these are very onesided tests and don't want to bias scores in list)
                self.maxprob = max(initprob, self.maxprob)
                if(initprob >0):
                    self.problist.append(initprob)  # add a very big bump in prob space so KL will see it
                    # print('Inital probability checks set prob to 1 with actual_state={}, next={}, dataval={}, '.format(actual_state,
                    #                                                                                expected_state,
                    #                                                                                data_val))                          
                          

            if (False and self.debug):
                self.debugstring = 'Early Instance: actual_state={}, next={}, dataval={}, '.format(actual_state,expected_state,data_val)
            return action
        else:


            data_val = self.format_data(data_val)
            prob_values = []
            actual_state = self.format_data(actual_state)
            difference_from_expected = data_val - actual_state  # next 4 are the difference between expected and actual state after one step, i.e.
            current = difference_from_expected


            probability = self.cstate_diff_prob(current)
            
                                                                                                   
#             # if diff is almost floatingpoint zero no point in computing EVM probbility, which will be 0 (tested to 1e-15)
#             valavg = np.array([                2.47088169e-05, 2.74201900e-05, 0.00000000e+00, 5.78351166e-06,       8.42652875e-04, 0.00000000e+00, 9.31653723e-06, 1.63258612e-05,       1.44448331e-05, 3.01137170e-06, 4.46659141e-04, 1.21615485e-03,       1.23267419e-04])

#             valstd = 10*np.array([3.11909380e-04, 5.47332869e-04, 1.00000000e-6, 1.77351837e-04,        5.33705301e-02,  1.00000000e-06, 2.65225600e-04, 4.46044523e-04,        5.83986090e-04, 3.73722490e-04, 1.43145027e-03, 5.87269744e-02,        3.76467718e-03])

# #            pdb.set_trace()
#             mval = max(abs(current) )
#             if (mval < 1e-4):
#                 probability = 0
#             else:
#                 # compute gusssian  proabilties for now  replace next 3 lines with EVT calc

#                 zscores = abs((valavg - current)/valstd)
#                 prob_values = [(2*st.norm.cdf(z)-1) for z in zscores]  # this is like evt w-score 
#                 probability = max(prob_values)   #take max over dimensions of probability of unkown
#                 if(probability > .5):
#                     maxindex = np.argmax(prob_values)
#                     self.character += 'Delta[{}] prob {}  @ step {};'.format(maxindex,probability,self.cnt)
#                 del zscores

                # data_tensor = torch.from_numpy(np.asarray(current))
                # probs = self.evm_inference_obj(data_tensor)
                # probability = self.prob_scale * (
                # probs.numpy()[0]) - 1  # probably of novelty so knowns have prob 0,  unknown prob 1.
            self.maxprob = max(probability, self.maxprob)
            # we can also include the score from control algorithm,   we'll have to test to see if it helps..
            #first testing suggests is not great as when block interfer it raises score as we try to fix it but then it seems novel. 
            #                self.maxprob=min(1,self.maxprob +  self.env_prediction.lastscore / self.scalelargescores)
            if (self.cnt > 6 and len(self.problist)>0 ):
                self.meanprob = np.mean(self.problist)

#            if (self.debug):
#                self.debugstring = 'Instance: cnt={},actual_state={}, next={},  current/diff={},NovelProb={}'.format(
#                    self.cnt, actual_state, expected_state, current, probability)
#                self.debugstring  +=           print("prob/probval", probability, prob_values, "maxprob", self.maxprob, "meanprob", self.meanprob,  " current", current)                
            # if(probability > .5):
            #     print("prob/maxval", probability, mval, "maxprob", self.maxprob, "meanprob", self.meanprob )
            #     print( " diff ", current)                                
            #     print("  exected", data_val)
            #     print("  actual", actual_state)                
                
            #          elif(self.given): self.statelist.append([action,actual_state,expected_state,current])
#            del prob_values

            if(self.given):
                probability=1
                self.env_prediction.lastscore  = self.env_prediction.lastscore *10    #make the scores highrer so we tend to use the more exensive twostep                 
            

            self.problist.append(probability)
        return action
