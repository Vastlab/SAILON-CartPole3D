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
#import cv2
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
import data_loader as DATA


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
        # we only use the first sets of scores for KL because novels worlds close when balanced
        self.scoreforKL = 20
        self.num_epochs = 200
        self.num_dims = 4
        self.num_blocks = None
        self.scalelargescores = 20
        # takes a while for some randome starts to stabilise so don't reset too early as it
        # reduces world change sensitvity.  Effective min is 1 as need at least a prior state to get prediction.
        self.skipfirstNscores = 1
        # both max for per episode individual prob as well as prob scale.
        self.maxinitprob = 4
        self.current_state = None
#        self.statelist=np.empty(200, dtype=object)

        self.blockmin = 999
        self.blockmax = -999
        self.blockvelmax = -999
        self.normblockmin = 999
        self.normblockmax = -999

        # if we see this many in a row we declare world changed as we never see even 3 in training
        self.maxconsecutivefailthresh = 4

        # we penalize for high failure rantes..  as  difference (faildiff*self.failscale) )
        self.failscale = 8.0  # How we scale failure fraction.. can be larger than one since its fractional differences and genaerally < .1 mostly < .05
        # Max fail fraction,  when above  this we start giving world-change probability for  failures
        self.failfrac = .25

        # because of noisy simulatn and  many many fields and its done each time step, we limit how much this can add per time step
        self.maxdynamicprob = .175
        # because of broken simulator we get randome bad value in car/velocity. when we detect them we limit their impact to this ..
        self.maxclampedprob = .005
        self.clampedprob = self.maxclampedprob
        # we scale prob from cart/pole because the environmental noise, if we fix it this will make it easire to adapt .
        self.cartprobscale = .25
        # we scale prob from initial state by this amount (scaled as consecuriteinit increases) and add world accumulator each time. No impacted by blend this balances risk from going of on non-novel worlds
        self.initprobscale = 1.0
        self.consecutiveinit = 0   # if get consecutitve init failures we keep increasing scale
        self.dynamiccount = 0   # if get consecutitve dynamic failures we keep increasing scale
        # if get consecutitve world change overall we keep increasing scale
        self.consecutivewc = 0

        # Large "control scores" often mean things are off, since we never know the exact model we reset when scores get
        # too large in hopes of  better ccotrol
        self.scoretoreset = 1000

        # smoothed performance plot for dtection.. see perfscore.py for compuation.  Major changes in control mean these need updated
        self.perflist = []
        self.mean_perf = 0.8883502538071065
        self.stdev_perf = 0.0824239133691708
        # How much do we weight Performacne KL prob.  make this small since it is slowly varying and added every episode. Small is  less sensitive (good for FP avoid, but yields slower detection).
        self.PerfScale = 0.15

        self.consecutivesuccess = 0
        self.consecutivefail = 0
        self.maxconsecutivefail = 0
        self.maxconsecutivesuccess = 0
        self.minprob_consecutive = .1
        self.mindynprob = .01
        self.dynblocksprob = 0
        # should be not be larger than maxdynamicprob
        assert (self.minprob_consecutive <= self.maxdynamicprob)
        self.tick = 0

        self.maxcarlen = 25600

        # TODO: change evm data dimensions
        if (self.num_dims == 4):
            self.mean_train = 0
            self.stdev_train = 0.0
            # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
            self.dynam_prob_scale = 2
        else:
            # self.mean_train = .198   #these are old values from Phase 1 2D cartpole..  for Pahse 2 3D we compute frm a training run.
            #self.stdev_train = 0.051058052318592555
            self.mean_train = 0.004
            self.stdev_train = 0.009
        # probably do need to scale but not tested sufficiently to see what it needs.
        self.dynam_prob_scale = 1

        self.cnt = 0
        self.framecnt = 0
        self.saveframes = False
        self.saveprefix = random.randint(1, 10000)

        self.worldchanged = 0
        self.worldchangedacc = 0
        self.previous_wc = 0
        # fraction of new prob we use when blending up..  It adapts over time
        self.blenduprate = 1
        # fraction of new prob we use when blending down..  should be less than beld up rate.  No use of max
        self.blenddownrate = .1

        self.failcnt = 0
        self.worldchangeblend = 0
        # from WSU "train".. might need ot make this computed.
        #self.mean_train=  0.10057711735799268
        #self.stdev_train = 0.00016
        self.problist = []
        self.scorelist = []
        self.maxprob = 0
        self.meanprob = 0
        self.noveltyindicator = None
        self.correctcnt = 0
        self.rcorrectcnt = 0
        self.totalcnt = 0
        self.perf = 0
        self.perm_search = 0
        self.prev_action = 0
        self.prev_state = None
        self.prev_predict = None

        #self.expected_backtwo = np.zeros(4)
        self.episode = 0
        self.trial = 0
        self.given = False

        #self.statelist=np.empty(200, dtype=object)
        self.debug = False
        self.debug = True
        self.debugstring = ""
        self.logstr = ""
        self.summary = ""
        self.hint = ""
        self.trialchar = ""

        self.imax = self.imin = self.imean = self.istd = self.ishape = self.iscale = None
        self.dmax = self.dshape = self.dscale = self.dmean = self.dstd = None

        if (False):
            # self.uccscart.reset(actual_state)
            ldist = self.line_to_line_dist(np.array([0, 0, 1]), np.array(
                [0, 0, -1]), np.array([0, 1, 0]), np.array([0, -1, 0]))
            ldist2 = self.line_to_line_dist(np.array([0, 0, 1]), np.array(
                [0, 0, -1]), np.array([0, 1, 0]), np.array([0, -1, .001]))
            pdist = self.point_to_line_dist(
                np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 1, 0]))
            pdist2 = self.point_to_line_dist(
                np.array([0, 0, 0]), np.array([0, 2, 0]), np.array([0, 1, 0]))
            print("line distances", ldist, ldist2, "Pointdist", pdist, pdist2)

        # Create prediction environment
        env_location = importlib.util.spec_from_file_location('CartPoleBulletEnv',
                                                              'cartpolepp/UCart.py')
        env_class = importlib.util.module_from_spec(env_location)
        env_location.loader.exec_module(env_class)

        myconfig = dict()
        myconfig['start_zeroed_out'] = False

        # Package params here
        params = dict()
        params['seed'] = 0
        params['config'] = myconfig
        #params['path'] = "WSU-Portable-Generator/source/partial_env_generator/envs/cartpolepp"
        params['path'] = "./cartpolepp        "
        params['use_img'] = False
        params['use_gui'] = False

        self.uccscart = env_class.CartPoleBulletEnv(params)
        self.uccscart.path = "./cartpolepp"

        self.starttime = datetime.now()
        self.cumtime = self.starttime - datetime.now()
        return


def reset(self, episode):
    self.problist = []
    self.scorelist = []
#        self.statelist=np.empty(200, dtype=object)
    self.given = False
    self.maxprob = 0
    self.meanprob = 0
    self.cnt = 0
    self.logstr = ""
    self.debugstring = ""
    self.episode = episode
    self.worldchanged = 0
    self.uccscart.resetbase()
    self.uccscart.reset()
    self.uccscart.episode = episode
    self.dynblocksprob = 0
    self.dynamiccount = 0

    if (episode == 0):
        self.blockmin = 999
        self.blockmax = -999
        self.blockvelmax = -999
        self.normblockmin = 999
        self.normblockmax = -999

    if (episode < 10):
        self.normblockmin = min(self.blockmin, self.normblockmin)
        self.normblockmax = max(self.blockmax, self.normblockmax)
        if (self.uccscart.tbdebuglevel > 1):
            print("PreNovelty Episode" + str(episode) + "Minmax block norms",
                  self.blockmin, self.blockmax, self.normblockmin, self.normblockmax)
    # reset things that we carry over between episodes withing the same trial.. but also need at least enough for KL
    elif (episode < self.scoreforKL):
        self.worldchangedacc = 0
        self.uccscart.wcprob = 0
        self.failcnt = 0
        self.worldchangeblend = 0
        self.consecutivefail = 0
        self.perm_search = 0
        self.trialchar = ""
        self.uccscart.characterization = {
            'level': None, 'entity': None, 'attribute': None, 'change': None}
        # don't use the noisy stuff much at very begining, to avoid FP going early
        self.clampedprob = self.maxclampedprob/2
        # fraction of new prob we use when blending up..  It adapts over time
        self.blenduprate = 1
        self.blockmin = 999
        self.blockmax = -999
        self.blockvelmax = -999
    # long term the noisy/bad probabilities seem to grow so reduce the max they can impact
    elif (episode < 2*self.scoreforKL):
        # we reduce max from noisy ones over the window size
        self.clampedprob = self.maxclampedprob * \
            ((2*self.scoreforKL-episode)/(self.scoreforKL))**2
        # fraction of new prob we use when blending up..  It adapts over time
        self.blenduprate = 1
        self.blockmin = 999
        self.blockvelmax = -999
        self.blockmax = -999
    else:
        self.clampedprob = 0
        # fraction of new prob we use when blending up..  It adapts over time
        self.blenduprate = max(.1, (3*self.scoreforKL -
                               episode)/(self.scoreforKL))
        self.blenddownrate = min(.5, self.blenddownrate + .05)

    if (episode > 3*self.scoreforKL):  # stop blending once we have stable KL values, and don't search since its expensive but cannot be useful after that many
        self.failcnt = 0
        self.worldchangeblend = 0
        self.failcnt = 0
        self.consecutivefail = 0
        self.perm_search = 0

    if (self.worldchangedacc > .5):
        self.uccscart.wcprob = self.worldchangedacc

    def mark_tried_actions(self, action, paction):
        '''  mark permutaiton indicies where the current action is in the perturbed action slot has been tried and did not work (used) '''
        for index in range(len(self.uccscart.actions_plist)-1):
            if (self.uccscart.actions_plist[index][action] == paction):
                self.uccscart.actions_permutation_tried[self.uccscart.actions_permutation_index] = 1

    def try_actions_permutations(self, actual_state, diffprob):
        ''' try various permuation of actions to see if they best explain the current state.  If it finds one return prob 1 and sets action permutation index in UCCScart.
         this can be called multiple times because the actual state transition can only use 1 action so if we swap say left-right  we might not see if we need to also swap front/back
        If here action_history should be populated with all actions  
        '''

        return 0
# !!!!!#####  end domain dependent adaption
