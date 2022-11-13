# UCCS TA 2 helper

# from utils import rollout
# import cv2
# from my_lib import *

# from utils import rollout
# import cv2
import torch
import torch.multiprocessing as mp

from current_config import CurrentConfig

# from my_lib import *

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
        self.cur_conf = CurrentConfig()

    def reset(self, episode):
        self.cur_conf.problist = []
        self.cur_conf.scorelist = []
        #        self.statelist=np.empty(200, dtype=object)
        self.cur_conf.given = False
        self.cur_conf.maxprob = 0
        self.cur_conf.meanprob = 0
        self.cur_conf.cnt = 0
        self.cur_conf.logstr = ""
        self.cur_conf.debugstring = ""
        self.cur_conf.episode = episode
        self.cur_conf.worldchanged = 0
        self.cur_conf.uccscart.resetbase()
        self.cur_conf.uccscart.reset()
        self.cur_conf.uccscart.episode = episode
        self.cur_conf.dynblocksprob = 0
        self.cur_conf.dynamiccount = 0

        if episode == 0:
            self.cur_conf.blockmin = 999
            self.cur_conf.blockmax = -999
            self.cur_conf.blockvelmax = -999
            self.cur_conf.normblockmin = 999
            self.cur_conf.normblockmax = -999

        if episode < 10:
            self.cur_conf.normblockmin = min(self.cur_conf.blockmin, self.cur_conf.normblockmin)
            self.cur_conf.normblockmax = max(self.cur_conf.blockmax, self.cur_conf.normblockmax)
            if self.cur_conf.uccscart.tbdebuglevel > 1:
                print("PreNovelty Episode" + str(episode) + "Minmax block norms",
                      self.cur_conf.blockmin, self.cur_conf.blockmax, self.cur_conf.normblockmin,
                      self.cur_conf.normblockmax)
        # reset things that we carry over between episodes withing the same trial.. but also need at least enough for KL
        elif episode < self.cur_conf.scoreforKL:
            self.cur_conf.worldchangedacc = 0
            self.cur_conf.uccscart.wcprob = 0
            self.cur_conf.failcnt = 0
            self.cur_conf.worldchangeblend = 0
            self.cur_conf.consecutivefail = 0
            self.cur_conf.perm_search = 0
            self.cur_conf.trialchar = ""
            self.cur_conf.uccscart.characterization = {
                'level': None, 'entity': None, 'attribute': None, 'change': None}
            # don't use the noisy stuff much at very begining, to avoid FP going early
            self.cur_conf.clampedprob = self.cur_conf.maxclampedprob / 2
            # fraction of new prob we use when blending up..  It adapts over time
            self.cur_conf.blenduprate = 1
            self.cur_conf.blockmin = 999
            self.cur_conf.blockmax = -999
            self.cur_conf.blockvelmax = -999
        # long term the noisy/bad probabilities seem to grow so reduce the max they can impact
        elif episode < 2 * self.cur_conf.scoreforKL:
            # we reduce max from noisy ones over the window size
            self.cur_conf.clampedprob = self.cur_conf.maxclampedprob * \
                                        ((2 * self.cur_conf.scoreforKL - episode) / (self.cur_conf.scoreforKL)) ** 2
            # fraction of new prob we use when blending up..  It adapts over time
            self.cur_conf.blenduprate = 1
            self.cur_conf.blockmin = 999
            self.cur_conf.blockvelmax = -999
            self.cur_conf.blockmax = -999
        else:
            self.cur_conf.clampedprob = 0
            # fraction of new prob we use when blending up..  It adapts over time
            self.cur_conf.blenduprate = max(.1, (3 * self.cur_conf.scoreforKL -
                                                 episode) / (self.cur_conf.scoreforKL))
            self.cur_conf.blenddownrate = min(.5, self.cur_conf.blenddownrate + .05)

        if episode > 3 * self.cur_conf.scoreforKL:  # stop blending once we have stable KL values, and don't search since its expensive but cannot be useful after that many
            self.cur_conf.failcnt = 0
            self.cur_conf.worldchangeblend = 0
            self.cur_conf.failcnt = 0
            self.cur_conf.consecutivefail = 0
            self.cur_conf.perm_search = 0

        if self.cur_conf.worldchangedacc > .5:
            self.cur_conf.uccscart.wcprob = self.cur_conf.worldchangedacc

        def mark_tried_actions(self, action, paction):
            '''  mark permutaiton indicies where the current action is in the perturbed action slot has been tried and did not work (used) '''
            for index in range(len(self.cur_conf.uccscart.actions_plist) - 1):
                if (self.cur_conf.uccscart.actions_plist[index][action] == paction):
                    self.cur_conf.uccscart.actions_permutation_tried[
                        self.cur_conf.uccscart.actions_permutation_index] = 1

        def try_actions_permutations(self, actual_state, diffprob):
            ''' try various permuation of actions to see if they best explain the current state.  If it finds one return prob 1 and sets action permutation index in UCCScart.
             this can be called multiple times because the actual state transition can only use 1 action so if we swap say left-right  we might not see if we need to also swap front/back
            If here action_history should be populated with all actions
            '''

            return 0
    # !!!!!#####  end domain dependent adaption
