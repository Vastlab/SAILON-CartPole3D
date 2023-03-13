#!/usr/bin/env python3
# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON TA2 Agent Example                                                           ** #
# **                                                                                            ** #
# **        Brian L Thomas, 2020                                                                ** #
# **                                                                                            ** #
# **  Tools by the AI Lab - Artificial Intelligence Quotient (AIQ) in the School of Electrical  ** #
# **  Engineering and Computer Science at Washington State University.                          ** #
# **                                                                                            ** #
# **  Copyright Washington State University, 2020                                               ** #
# **  Copyright Brian L. Thomas, 2020                                                           ** #
# **                                                                                            ** #
# **  All rights reserved                                                                       ** #
# **  Modification, distribution, and sale of this work is prohibited without permission from   ** #
# **  Washington State University.                                                              ** #
# **                                                                                            ** #
# **  Contact: Brian L. Thomas (bthomas1@wsu.edu)                                               ** #
# **  Contact: Larry Holder (holder@wsu.edu)                                                    ** #
# **  Contact: Diane J. Cook (djcook@wsu.edu)                                                   ** #
# ************************************************************************************************ #
import copy
import optparse
import queue
import random
import threading
import time

from objects.TA2_logic import TA2Logic

import pdb
import torch
import numpy as np
import UCCS_TA2_helper
from UCCS_TA2_helper import UCCSTA2
import uuid
import csv

from datetime import datetime, timedelta



#import tracemalloc

#tracemalloc.start(10)
#snapshot1 = snapshot2 = tracemalloc.take_snapshot()
#top_stats = snapshot1.statistics('lineno')





class ThreadedProcessingExample(threading.Thread):
    def __init__(self, processing_object: list, response_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.processing_object = processing_object
        self.response_queue = response_queue
        self.is_done = False
        return

    def run(self):
        """All work tasks should happen or be called from within this function.
        """
        is_novel = False
        message = ''

        # Do some fake work here.
        for work in self.processing_object:
            sum = 0
            for i in range(work):
                sum += i
            message += 'Did sum of {}. '.format(work)

        self.response_queue.put((is_novel, message))
        return

    def stop(self):
        self.is_done = True
        return


class TA2Agent(TA2Logic):
    def __init__(self, options):
        super().__init__()

        self.possible_answers = list()
        # This variable can be set to true and the system will attempt to end training at the
        # completion of the current episode, or sooner if possible.
        #        self.end_training_early = False
        self.end_training_early = True

        # This variable is checked only during the evaluation phase.  If set to True the system
        # will attempt to cleanly end the experiment at the conclusion of the current episode, 
        # or sooner if possible.
        self.end_experiment_early = False

        # Define evm and pilco objects as well as state data list
        self.totalSteps = 0
        self.lasttime = 0
        self.UCCS = UCCSTA2()
        self.UCCS.debug = options.debug
        self.UCCS.debug = True
        # Self states, always previous 4 steps

        return

    def experiment_start(self):
        """This function is called when this TA2 has connected to a TA1 and is ready to begin
        the experiment.
        """
        self.log.info('Experiment Start')
        return

    def training_start(self):
        """This function is called when we are about to begin training on episodes of data in
        your chosen domain.
        """
        self.log.info('Training Start')
        return

    def training_episode_start(self, episode_number: int):
        """This function is called at the start of each training episode, with the current episode
        number (0-based) that you are about to begin.
        Parameters
        ----------
        episode_number : int
            This identifies the 0-based episode number you are about to begin training on.
        """
        self.log.info('Training Episode Start: #{}'.format(episode_number))
        self.UCCS.reset(episode_number)
        return

    def training_instance(self, feature_vector: dict, feature_label: dict) -> dict:
        """Process a training
        Parameters
        ----------
        feature_vector : dict
            The dictionary of the feature vector.  Domain specific feature vector formats are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        feature_label : dict
            The dictionary of the label for this feature vector.  Domain specific feature labels
            are defined on the github (https://github.com/holderlb/WSU-SAILON-NG). This will always
            be in the format of {'action': label}.  Some domains that do not need an 'oracle' label
            on training data will receive a valid action chosen at random.
        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """

        #        self.log.debug('Training Instance: feature_vector={}  feature_label={}.format(
        #            feature_vector, feature_label))


        action = self.UCCS.process_instance(feature_vector)
        #        if(self.UCCS.episode == 0 and  self.UCCS.cnt <20):
        #            self.log.debug(self.UCCS.debugstring)
        self.totalSteps += 1
        # format the return of novelty and actions
        action = {"action": action}

        return action

    def training_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.
        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        #        self.log.debug('Training Performance: {}'.format(performance))
        return

    def training_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        """Provides the final performance on the training episode and indicates that the training
        episode has ended.
        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
#        novelty_probability = self.UCCS.world_change_prob(True)   # learn prob from train
        novelty_probability = self.UCCS.world_change_prob(False)   #one bad start will break it so we use pretrained data insteadb     
        novelty_threshold = 0.5
        novelty = 0
        novelty_characterization = dict()

        #        print("Novelty Probability:", novelty_probability)
        #        print("Total Steps:", self.totalSteps)
        self.log.debug('Training Episode End: performance={}, NovelProb={}, steps={}, WC={}, '.format(performance, 
                                                                                                    self.UCCS.problist,
                                                                                                    self.totalSteps,
                                                                                                    novelty_probability))
        self.log.info('Training Episode End: steps={}, WC={}, '.format(self.totalSteps, novelty_probability))
        self.totalSteps = 0
        return novelty_probability, novelty_threshold, novelty, novelty_characterization

    def training_end(self):
        """This function is called when we have completed the training episodes.
        """
        self.log.info('Training End')
        return

    def train_model(self):
        """Train your model here if needed.  If you don't need to train, just leave the function
        empty.  After this completes, the logic calls save_model() and reset_model() as needed
        throughout the rest of the experiment.
        """

        return

    def save_model(self, filename: str):
        """Saves the current model in memory to disk so it may be loaded back to memory again.
        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        self.log.info('Save model to disk.')
        return

    def reset_model(self, filename: str):
        """Loads the model from disk to memory.
        Parameters
        ----------
        filename : str
            The filename where the model was stored.
        """
        self.log.info('Load model from disk.')
        return

    def trial_start(self, trial_number: int, novelty_description: dict):
        """This is called at the start of a trial with the current 0-based number.
        Parameters
        ----------
        trial_number : int
            This is the 0-based trial number in the novelty group.
        novelty_description : dict
            A dictionary that will have a description of the trial's novelty.
        """
        self.log.info('Trial Start: #{}  novelty_desc: {}'.format(trial_number, 
                                                                  str(novelty_description)))
        self.UCCS.trial = trial_number
        '''if len(self.possible_answers) == 0:
            self.possible_answers.append(dict({'action': 'left'}))'''
        return

    def testing_start(self):
        """This is called after a trial has started but before we begin going through the
        episodes.
        """
        self.log.info('Testing Start')
        return

    def testing_episode_start(self, episode_number: int):
        """This is called at the start of each testing episode in a trial, you are provided the
        0-based episode number.
        Parameters
        ----------
        episode_number : int
            This is the 0-based episode number in the current trial.
        """
        self.log.info('Testing Episode Start: #{}'.format(episode_number))
        self.UCCS.reset(episode_number)
        # if(True or self.UCCS.debug):
        #     snapshot1 = tracemalloc.take_snapshot()

        self.UCCS.starttime = datetime.now()
        return

        # One step predictions here

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> dict:
        """Evaluate a testing instance.  Returns the predicted label or action, if you believe
        this episode is novel, and what novelty level you beleive it to be.
        Parameters
        ----------
        feature_vector : dict
            The dictionary containing the feature vector.  Domain specific feature vectors are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        novelty_indicator : bool, optional
            An indicator about the "big red button".
                - True == novelty has been introduced.
                - False == novelty has not been introduced.
                - None == no information about novelty is being provided.
        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """

        self.UCCS.noveltyindicator = novelty_indicator
        self.UCCS.debugstring = ""
#        self.log.debug('Testing Instance: feature_vector={}'.format(feature_vector))
        
        if(self.UCCS.cnt < 1):
#            self.UCCS.hint=str(feature_vector['hint'])
            self.UCCS.hint=""
#            self.log.debug( 'Epi {}, Hint={}, nno={}'.format(self.UCCS.uccscart.episode,feature_vector['hint'], novelty_indicator))
            self.log.debug(
                'Testing Instance: feature_vector={}, novelty_indicator={}'.format(feature_vector, novelty_indicator))

            if (novelty_indicator == True):
                self.UCCS.given = True
                self.UCCS.uccscart.givendetection = True                
            else:
                self.UCCS.given = False
                self.UCCS.uccscart.givendetection = False                                


        action = self.UCCS.process_instance(feature_vector)

        #        if(self.UCCS.episode == 0 and  self.UCCS.cnt > 101 and  self.UCCS.cnt <106):#
        #           self.log.debug(self.UCCS.debugstring)
        #        if(self.UCCS.episode == 0 and  self.UCCS.cnt <120):
        self.log.debug(self.UCCS.debugstring)
        self.totalSteps += 1


        # format the return of novelty and actions
        action = {"action": action}
        return action

    def testing_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.
        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """

        #        self.log.info('Test Performance: {}'.format(performance))

        return

    def testing_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        """Provides the final performance on the testing episode.
        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """

        global TA2_previous_worldchange   # global so it retains info between trials        
        if(self.UCCS.episode <1):
            TA2_previous_worldchange = 0
        
        novelty_probability = self.UCCS.world_change_prob(False)
        if(novelty_probability < TA2_previous_worldchange):
            self.log.info("??? Unknown Error WorldChange whent down at episode {} from {} to {}.  Resetting".format(self.UCCS.episode,TA2_previous_worldchange,novelty_probability))
            novelty_probability =  TA2_previous_worldchange
            self.UCCS.worldchangeacc = TA2_previous_worldchange
        else:
            TA2_previous_worldchange = novelty_probability
            
                     
        novelty_threshold = 0.5
        novelty = 0
        novelty_characterization =  self.UCCS.uccscart.characterization
        novelty_characterization['summary'] = self.UCCS.summary        
#        novelty_characterization['stringdump'] = self.UCCS.logstr


        end =datetime.now()
        self.UCCS.cumtime += end - self.UCCS.starttime
        

        self.UCCS.totalcnt += 1
        self.UCCS.perf += performance
        self.UCCS.perflist.append(performance)
        
        rcorrect = pcorrect = 0
        if(performance > .99): rcorrect = 1.0
        iscorrect = 0
        if((self.UCCS.noveltyindicator == True) and (novelty_probability >= .5)): iscorrect = 1
        if((self.UCCS.noveltyindicator == False) and (novelty_probability < .5)): iscorrect = 1
        self.UCCS.correctcnt = self.UCCS.correctcnt + iscorrect
        self.UCCS.rcorrectcnt = self.UCCS.rcorrectcnt + max(iscorrect, rcorrect)
        pcorrect = 100*self.UCCS.correctcnt/(self.UCCS.totalcnt)
        rperf = 100*self.UCCS.perf/(self.UCCS.totalcnt)        
        self.log.info('Testing End#={}: WC={} Hint={} steps={}, CPerf={} times={} {}  NovI={}  Cor={}, Rcor={}, pco={}, CCnt={}, RCCnt={} TCN={}, Char={}  '.format(
            self.UCCS.episode,
            round(novelty_probability, 5),
            self.UCCS.hint,
            self.totalSteps,            
            round(rperf, 1),            
            round((end - self.UCCS.starttime).total_seconds(), 1), round((self.UCCS.cumtime/self.UCCS.totalcnt).total_seconds(), 1), 
            self.UCCS.noveltyindicator,     iscorrect, rcorrect, round(pcorrect, 2),
            self.UCCS.correctcnt, self.UCCS.rcorrectcnt, self.UCCS.totalcnt, 
            str(novelty_characterization)
        ))


        self.log.debug('DTend# {}  Logstr={} {} Prob={} {} Scores= {}   '.format( self.UCCS.episode,  self.UCCS.logstr,
            "\n", [round(num, 2) for num in self.UCCS.problist[:40]], 
            "\n", [round(num, 2) for num in self.UCCS.scorelist[:40]]))            

        self.totalSteps = 0
        
        
        self.UCCS.starttime = datetime.now()
        # if(self.UCCS.debug):
        #     snapshot2 = tracemalloc.take_snapshot()

        #     current, peak = tracemalloc.get_traced_memory()
        #     print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        # if(self.UCCS.debug):
        #     top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #     print("[ Top 20 differences ]")
        #     for stat in top_stats[:20]:
        #         print(stat)


        
        # if(self.UCCS.given):
        #      fname = 'Given-History-{}-{}-{}.csv'.format(self.UCCS.trial, self.UCCS.episode, uuid.uuid4().hex)
        #      with open(fname, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerows(self.UCCS.statelist)
        #         f.close()
        #         self.UCCS.given=False

        return novelty_probability, novelty_threshold, novelty, novelty_characterization

    def testing_end(self):
        """This is called after the last episode of a trial has completed, before trial_end().
        """
        self.log.info('Testing End')
        return

    def trial_end(self):
        """This is called at the end of each trial.
        """
        self.log.info('Trial End')
        return

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        return


if __name__ == "__main__":
    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--config", 
                      dest="config", 
                      help="Custom ClientAgent config file.", 
                      default="TA2.config")
    parser.add_option("--debug", 
                      dest="debug", 
                      action="store_true", 
                      help="Set logging level to DEBUG from INFO.", 
                      default=False)
    parser.add_option("--fulldebug", 
                      dest="fulldebug", 
                      action="store_true", 
                      help="Set logging level to DEBUG from INFO for all imported libraries.", 
                      default=False)
    parser.add_option("--logfile", 
                      dest="logfile", 
                      help="Filename if you want to write the log to disk.")
    parser.add_option("--printout", 
                      dest="printout", 
                      action="store_true", 
                      help="Print output to the screen at given logging level.", 
                      default=False)
    parser.add_option("--no-testing", 
                      dest="no_testing", 
                      action="store_true", 
                      help=('Instruct the TA2 to just create the experiment, update the config, '
                            'consume training data (if any), train the model (if needed), saves '
                            'the model to disk, and then exits. This disables the use of '
                            '--just-one-trial when set.'), 
                      default=False)
    parser.add_option("--just-one-trial", 
                      dest="just_one_trial", 
                      action="store_true", 
                      help="Process just one trial and then exit.", 
                      default=False)
    parser.add_option("--ignore-secret", 
                      dest="ignore_secret", 
                      action="store_true", 
                      help='Causes the program to ignore any secret stored in experiment_secret.', 
                      default=False)
    (options, args) = parser.parse_args()
    if options.fulldebug:
        options.debug = True

    agent = TA2Agent(options)
    agent.run()
