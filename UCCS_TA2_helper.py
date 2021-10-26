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
from vast.opensetAlgos.EVM import EVM_Training, EVM_Inference, EVM_Inference_cpu_max_knowness_prob
from vast import activations
from statistics import mean
import scipy.stats as st

import gc
import random
import csv
import importlib.util

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
        self.scoreforKl=20        # we only use the first sets of scores for KL because novels worlds close when balanced
        self.num_epochs = 200
        self.num_dims = 4
        self.scalelargescores=20
        # takes a while for some randome starts to stabilise so don't reset too early as it
        # reduces world change sensitvity        
        self.skipfirstNscores=3

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
        # from WSU "train".. might need ot make this computed.
        #        self.mean_train=  0.10057711735799268
        #       self.stdev_train = 0.00016
        self.problist = []
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

            # TODO: change filenames for new trained evm
            filename = "evm_models/evm_cosine_cartpole_tail_5000_ct_0.7_dm_0.55_largerdata.pkl"


            evm_model = pickle.load(open(filename, "rb"))
            self.evm_inference_obj = EVM_Inference_cpu_max_knowness_prob(args_evm.distance_metric, evm_model)
        return

    def reset(self, episode):
        self.problist = []
        self.statelist = []
        self.given = False
        self.maxprob = 0
        self.meanprob = 0
        self.cnt = 0
        self.episode = episode
        self.worldchanged = 0
        self.env_prediction.resetbase()
        self.env_prediction.reset()

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

    def world_change_prob(self,settrain=False):
        mlength = len(self.problist)
        mlength = min(self.scoreforKl,mlength)
        # we look at the larger of the begging or end of list.. world changes most obvious at the ends. 
#        mu = max(np.mean(self.problist[:mlength]), np.mean(self.problist[-mlength:]))
#        sigma = max(np.std(self.problist[:mlength]), np.std(self.problist[-mlength:]))
        mu = np.mean(self.problist[:mlength])
        sigma = np.std(self.problist[:mlength])
        if(settrain):
           self.mean_train = mu;
           self.stdev_train = sigma;
           print("Set  world change train mu and sigma", mu, sigma)
           self.worldchanged = 0
           return 0;
        if( self.mean_train == 0):
           self.mean_train = 0.052   #these guessted values for Phase 2 incase we get called without training
           self.stdev_train = 0.091
           self.prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.
           

#        if (len(self.problist) < 3):
#            print("Very short, world must have changed")
#            return 1;
        if (len(self.problist) < 198):   #for real work
            if (True or self.debug):
                self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)                
                self.debugstring = '   ***Short World Change Prob ={},mu={}, sigmas {}, mean {} stdev{} val {} thresh {}'.format(
                self.problist, mu, sigma, self.mean_train, self.stdev_train ,self.KL_val, self.KL_threshold)
                print(self.debugstring)
        if (sigma == 0):
            if (mu == self.mean_train):
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
        if (True or self.debug):
                self.debugstring = '      World Change Prob ={},mu={}, sigmas {}, mean {} stdev{} val {} thresh {} scale {}'.format(
                prob, mu, sigma, self.mean_train, self.stdev_train ,self.KL_val, self.KL_threshold, KLscale)
                print(self.debugstring)
        
        #        self.worldchanged = max(prob,self.worldchanged)
        self.worldchanged = prob
        return self.worldchanged

    def process_instance(self, actual_state):
        #        pertub = (self.cnt > 100) and (self.maxprob < .5)
        pertub = False
        forcedreset=False
#        pdb.set_trace()
        if(self.cnt ==0 ):
            observation = self.env_prediction.reset(actual_state)    #TB  its more sensitive to pertubations if we don't reset after first step

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
            if (False and self.debug):
                self.debugstring = 'Early Instance: actual_state={}, next={}, dataval={}, '.format(actual_state,
                                                                                                   expected_state,
                                                                                                   data_val)
            return action
        else:
            data_val = self.format_data(data_val)
            prob_values = []
            actual_state = self.format_data(actual_state)
            difference_from_expected = data_val - actual_state  # next 4 are the difference between expected and actual state after one step, i.e.
            current = difference_from_expected
            # if diff is almost floatingpoint zero no point in computing EVM probbility, which will be 0 (tested to 1e-15)
            valavg = np.array([                2.47088169e-05, 2.74201900e-05, 0.00000000e+00, 5.78351166e-06,       8.42652875e-04, 0.00000000e+00, 9.31653723e-06, 1.63258612e-05,       1.44448331e-05, 3.01137170e-06, 4.46659141e-04, 1.21615485e-03,       1.23267419e-04])

            valstd = 10*np.array([3.11909380e-04, 5.47332869e-04, 1.00000000e-6, 1.77351837e-04,        5.33705301e-02,  1.00000000e-06, 2.65225600e-04, 4.46044523e-04,        5.83986090e-04, 3.73722490e-04, 1.43145027e-03, 5.87269744e-02,        3.76467718e-03])

#            pdb.set_trace()
            mval = max(abs(current) )
            if (mval < 1e-4):
                probability = 0
            else:
                # compute gusssian  proabilties for now  replace next 3 lines with EVT calc

                zscores = abs((valavg - current)/valstd)
                prob_values = [(2*st.norm.cdf(z)-1) for z in zscores]  # this is like evt w-score 
                probability = max(prob_values)   #take max over dimensions of probability of unkown
                del zscores

                # data_tensor = torch.from_numpy(np.asarray(current))
                # probs = self.evm_inference_obj(data_tensor)
                # probability = self.prob_scale * (
                # probs.numpy()[0]) - 1  # probably of novelty so knowns have prob 0,  unknown prob 1.
                self.maxprob = max(probability, self.maxprob)
                # we also include the score from control algorithm 
                self.maxprob=min(1,self.maxprob +  self.env_prediction.lastscore / self.scalelargescores)
                if (self.cnt > 6):
                    self.meanprob = np.mean(self.problist)
                #              print(self.meanprob, self.problist)

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
            del prob_values
            self.problist.append(probability)
        return action
