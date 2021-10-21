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
from vast.opensetAlgos import EVM_Training, EVM_Inference, EVM_Inference_cpu_max_knowness_prob
from vast import activations
from statistics import mean
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
        self.num_epochs = 200
        self.num_dims = 4

        # TODO: change evm data dimensions
        if (self.num_dims == 4):
            self.mean_train = 0
            self.stdev_train = 0.0
            self.prob_scale = 2  # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
        else:
            self.mean_train = .198
            self.stdev_train = 0.051058052318592555
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
        if(self.cnt <2):
            observation = env.reset(state_given)    #TB  its more sensitive to pertubations if we don't reset after first step
        action, next_action, expected_state = env.get_best_action(observation)
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

    def world_change_prob(self):
        mu = np.mean(self.problist)
        sigma = np.std(self.problist)
        if (len(self.problist) < 3): return 0;
        if (sigma == 0):
            if (mu == self.mean_train):
                return 0;
            else:
                self.worldchanged = 1;
                return self.worldchanged

            #        pdb.set_trace()
        self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)
        KLscale = (
                              self.num_epochs + 1 - self.episode / 2) / self.num_epochs  # decrease scale (increase sensitvity)  from start down to  1/2
        prob = min(1.0, KLscale * self.KL_val / (2 * self.KL_threshold))
        #        self.worldchanged = max(prob,self.worldchanged)
        self.worldchanged = prob
        return self.worldchanged

    def process_instance(self, actual_state):
        #        pertub = (self.cnt > 100) and (self.maxprob < .5)
        pertub = False
#        pdb.set_trace()
        action, expected_state = self.takeOneStep(actual_state, self.env_prediction, pertub)

        data_val = self.expected_backone
        self.expected_backone = expected_state
        self.cnt += 1
        if (self.cnt < 3):  # skip prob estiamtes for the first ones as we need history to get prediction
            if (self.debug):
                self.debugstring = 'Early Instance: actual_state={}, next={}, dataval={}, '.format(actual_state,
                                                                                                   expected_state,
                                                                                                   data_val)
            return action
        else:
            data_val = self.format_data(data_val)
            actual_state = self.format_data(actual_state)
            difference_from_expected = data_val - actual_state  # next 4 are the difference between expected and actual state after one step, i.e.
            current = difference_from_expected
            # if diff is almost floatingpoint zero no point in computing EVM probbility, which will be 0 (tested to 1e-15)
            valavg = np.array([2.47088169e-05, 2.74201900e-05, 0.00000000e+00, 5.78351166e-06,       8.42652875e-04, 0.00000000e+00, 9.31653723e-06, 1.63258612e-05,       1.44448331e-05, 3.01137170e-06, 4.46659141e-04, 1.21615485e-03,       1.23267419e-04])

            valstd = np.array([3.11909380e-05, 5.47332869e-05, 1.00000000e-10, 1.77351837e-04,        1.33705301e-03,  1.00000000e-10, 2.65225600e-05, 2.46044523e-05,        5.83986090e-05, 3.73722490e-05, 1.43145027e-03, 1.87269744e-03,        3.76467718e-03])

#            pdb.set_trace()
            if (max(abs(current)) < 1e-6):
                probability = 0
            elif (self.meanprob > .6):
                # hack for speed, no point recomputing if we already have a high mean probability as world changed detected with this much probability so cannot go back
                probability = self.meanprob
            else:
                # compute EVM proabilties
                if ( 0 < max((valavg-valstd) - abs(current))):
                    probability = .0001

                elif ( 0 < max((valavg) - abs(current))):
                    probability = .01
                elif ( 0 < max((valavg+valstd) - abs(current))):
                    probability = .25
                elif ( 0 < max((valavg+1.5*valstd) - abs(current))):
                    probability = .5
                elif ( 0 < max((valavg+2*valstd) - abs(current))):
                    probability = .68
                elif ( 0 < max((valavg+3*valstd) - abs(current))):
                    probability = .95
                else:
                          probability = .98


                # data_tensor = torch.from_numpy(np.asarray(current))
                # probs = self.evm_inference_obj(data_tensor)
                # probability = self.prob_scale * (
                # probs.numpy()[0]) - 1  # probably of novelty so knowns have prob 0,  unknown prob 1.
                self.maxprob = max(probability, self.maxprob)
                if (self.cnt > 6):
                    self.meanprob = np.mean(self.problist)
                #              print(self.meanprob, self.problist)

            if (self.debug):
                self.debugstring = 'Instance: cnt={},actual_state={}, next={},  current/diff={},NovelProb={}'.format(
                    self.cnt, actual_state, expected_state, current, probability)
                print("prob", probability, "maxprob", self.maxprob, "meanprob", self.meanprob,  " current", current)
            print("prob", probability, "maxprob", self.maxprob, "meanprob", self.meanprob,  " current", current)

            #          elif(self.given): self.statelist.append([action,actual_state,expected_state,current])

            self.problist.append(probability)
        return action
