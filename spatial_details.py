import numpy as np

import basic_operation as basic_op
import formatter as formatter
import probability_difference as prob_dif
from current_config import CurrentConfig


def block_pos(istate, blocknum):
    return istate[13 + blocknum * 6:13 + blocknum * 6 + 3]


def block_vel(istate, blocknum):
    return istate[13 + blocknum * 6 + 3:13 + blocknum * 6 + 6]


def cart_pos(istate):
    return istate[0:3]


def cart_vel(istate):
    return istate[3:6]


def pole_pos(istate):
    return istate[6:9]


def pole_vel(istate):
    return istate[9:12]


def unit_vector(vector):
    """
        Returns the unit vector of the vector  and if norm is too small  limit divsiion to avoid numeric instability
    """
    return vector / max(np.linalg.norm(vector), 1e-16)


def vector_angle(v1, v2):
    cur_config = CurrentConfig()
    """
        Returns the angle in radians between vectors 'v1' and 'v2'::
        >>> vector_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        >>> vector_angle((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> vector_angle((1, 0, 0), (1, 0, 0))
        0.0
    """
    v1 = unit_vector(v1)
    v2 = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def process_instance(oactual_state):
    cur_config = CurrentConfig()
    #        pertub = (self.cnt > 100) and (self.maxprob < .5)
    pertub = False
    cur_config.current_state = oactual_state  # mostly for debugging
    # self.statelist[self.cnt] = oactual_state  # save all states, used in testing to see if trajectory is dynamic or balistic,. also useful for debugging u

    if cur_config.saveframes:
        image = oactual_state['image']
    cur_config.logstr += cur_config.uccscart.char  # copy overy any information about collisions
    #        if(len(self.uccscart.char)>0): print("Process inst with char", self.uccscart.char)
    # if cart control detected something we start from that estiamte
    probability = cur_config.uccscart.wcprob

    #        if("CP in        self.uccscart.char):
    #            self.uccscart.lastscore = 0.001111; #  if we had a lot fo collision potential, ignore the score.
    cur_config.scorelist.append(cur_config.uccscart.lastscore)
    cur_config.uccscart.char = ""  # reset any information about collisions
    action, expected_state = basic_op.take_one_step(oactual_state, cur_config.uccscart, pertub)

    # we can now fill in previous history's actual
    if 0 <= cur_config.uccscart.force_action < 5:
        # self.uccscart.action_history[self.uccscart.force_action][1] = self.uccscart.format_data(oactual_state)
        cur_config.uccscart.action_history[cur_config.uccscart.force_action][1] = formatter.format_data(oactual_state)

    # we don't reset in first few steps because random start may be a bad position yielding large score
    # might be were we search for better world parmaters if we get time for that
    # TB.. this does not seem to be needed as reset hasppend when searching for best action
    #        if(self.cnt > self.skipfirstNscores and self.uccscart.lastscore > self.scoretoreset):
    #            print("At step ", self.cnt, "resettin to actual because of a large score", self.uccscart.lastscore)
    #            self.uccscart.reset(actual_state)

    data_val = cur_config.prev_predict
    cur_config.prev_predict = expected_state
    cur_config.prev_state = oactual_state
    cur_config.cnt += 1
    if cur_config.cnt == 1:  # if first run cannot check dynamics just initial state
        if cur_config.uccscart.tbdebuglevel > 0:
            cur_config.debugstring = 'Testing initial state for obvious world changes: actual_state={}, next={}, dataval={}, '.format(
                oactual_state,
                expected_state,
                data_val)
        initprob = prob_dif.istate_diff_EVT_prob(oactual_state)

        # update max and add if initprob >0 add list (if =0 itnore as these are very onesided tests and don't want to bias scores in list)
        cur_config.maxprob = max(initprob, cur_config.maxprob)
        if initprob > 0:
            # add a very big bump in prob space so KL will see it
            cur_config.problist.append(initprob)
            if cur_config.uccscart.tbdebuglevel > 0:
                print(
                    'Init probability checks set prob to 1 with actual_state={}, next={}, dataval={}, problist={}, '.format(
                        oactual_state,
                        expected_state,
                        data_val,
                        cur_config.problist))

            # if (self.debug):
            #     self.debugstring = 'Early Instance: actual_state={}, next={}, dataval={}, '.format(oactual_state,expected_state,data_val)
            cur_config.prev_action = action
            return action
    else:
        data_val = formatter.format_data(data_val)
        prob_values = []
        actual_state = formatter.format_data(oactual_state)

        # if sizes changed then we have different number of blocks.. and it must be novel
        if len(data_val) != len(actual_state):
            probability = 1.0
            cur_config.uccscart.characterization['level'] = int(8)
            cur_config.uccscart.characterization['entity'] = "Block"
            cur_config.uccscart.characterization['attribute'] = "quantity"
            if len(data_val) >= len(actual_state):
                if cur_config.logstr.count("LL8") > 1:  # if we had more than one chance its increasing
                    cur_config.uccscart.characterization['change'] = 'decreasing'
                else:
                    cur_config.uccscart.characterization['change'] = 'decrease'
            else:
                if cur_config.logstr.count("LL8") > 1:
                    cur_config.uccscart.characterization['change'] = 'increase'
                else:
                    cur_config.uccscart.characterization['change'] = 'increasing'
            cur_config.worldchangedacc = 1
            tstring = " & P2+ LL8: Blocks quantity " + \
                      str(cur_config.uccscart.characterization['change']) + " FV len " + str(
                len(data_val)) + " changed to " + str(len(actual_state))
            if cur_config.logstr.count("LL8") < 2:
                cur_config.logstr += str(tstring)
            print(tstring)

        else:
            # vectors can be subtracted
            # next 4 are the difference between expected and actual state after one step, i.e.
            difference_from_expected = data_val - actual_state
            current = difference_from_expected

            diffprobability = cur_config.dynam_prob_scale * \
                              prob_dif.cstate_diff_EVT_prob(current, actual_state)
            # blocks are less noisy so we always add them in
            probability += cur_config.dynblocksprob

            # if dynamics is really off, it should be off almost every time, so only use it when we have a large count and limit its growth, let KL detect it
            if cur_config.dynamiccount > 4 and diffprobability > 0:
                probability = min(cur_config.maxdynamicprob,
                                  probability + diffprobability)
                # if we don't see dyamics reduce count
                cur_config.dynamiccount = cur_config.dynamiccount + 1
                if cur_config.uccscart.tbdebuglevel > 0:
                    print(" In step " + str(cur_config.tick) + " usin dyn prob ",
                          cur_config.dynamiccount, diffprobability, probability)
            else:
                if diffprobability > .05 and cur_config.uccscart.tbdebuglevel > 0:
                    print(" Step " + str(cur_config.tick) + " skipping low dyncnt and its prob ",
                          cur_config.dynamiccount, diffprobability, probability)
                    # if we don't see dyamics reduce count
                    cur_config.dynamiccount = cur_config.dynamiccount + 1
            if diffprobability <= 0.05 and cur_config.dynamiccount > 2:
                # if we don't see dyamics reduce count
                cur_config.dynamiccount = cur_config.dynamiccount - 1

        # if we have high enough probability and failed often enough and have not searched for pertubations, try searching for action permuations
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

        # !!!!!#####  end GLUE/API code EVT-
        # !!!!!#####  Start domain dependent adaption

        # if we have not had a lot successess in a row (sign index is right)   and declared world changed and  we ahve enough failures then try another index
        if False and cur_config.maxconsecutivesuccess < 5 and cur_config.maxconsecutivefail > cur_config.maxconsecutivefailthresh and cur_config.consecutivefail > 3:
            # try the next permuation.. see if we can reduce the fail rate
            cur_config.uccscart.actions_permutation_index += 1
            if cur_config.uccscart.actions_permutation_index > (len(cur_config.uccscart.actions_plist) - 1):
                cur_config.uccscart.actions_permutation_index = 0
            cur_config.logstr += "#####? Too many failures.  Guessing actions were mapped/perturbed.. Now using pertubation "
            cur_config.logstr.join(
                map(str, cur_config.uccscart.actions_plist[cur_config.uccscart.actions_permutation_index]))
            cur_config.logstr += "if this is the last time you see this message and performance is now good then characterize this as the action permutation in placeof the  uncontrollable characateration provided after world change provided earlier #####?"
            print(cur_config.logstr)
            cur_config.consecutivefail = 0

        probability = min(1, probability)
        cur_config.problist.append(probability)

        cur_config.maxprob = max(probability, cur_config.maxprob)
        # we can also include the score from control algorithm,   we'll have to test to see if it helps..
        # first testing suggests is not great as when block interfer it raises score as we try to fix it but then it seems novel.
        #                self.maxprob=min(1,self.maxprob +  self.uccscart.lastscore / self.scalelargescores)
        if cur_config.cnt > 0 and len(cur_config.problist) > 0:
            cur_config.meanprob = np.mean(cur_config.problist)

        if cur_config.uccscart.tbdebuglevel > 2:
            cur_config.debugstring = 'Instance: cnt={},actual_state={}, next={},  current/diff={},NovelProb={}'.format(
                cur_config.cnt, actual_state, expected_state, current, probability)
            print("prob/probval", probability, prob_values,
                  "maxprob", cur_config.maxprob, "meanprob", cur_config.meanprob)

    cur_config.prev_action = action

    if cur_config.saveframes:
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
            # resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # org = (10, 30)
            # fontScale = .5
            # # Blue color in BGR
            # if(round(self.worldchangedacc,3) < .5):            color = (255, 0, 0)
            # else  :            color = (0,0,255)
            # thickness = 2
            # fname = '/scratch/tboult/PNG/{1}-Frame-{0:04d}.png'.format(self.framecnt,self.saveprefix)
            # wstring = 'E={4:03d}.{0:03d} RP={7:4.3f} WC={2:4.3f} P{1:3.2f} N={6:.1} C={5:.4},S={3:12.6}'.format(self.uccscart.tick,probability,self.worldchangedacc,self.uccscart.lastscore,self.episode,self.logstr[-4:], str(self.noveltyindicator),100*self.perf/(max(1,self.totalcnt))        )
            # outimage = cv2.putText(resized, wstring, org, font,
            #                        fontScale, color, thickness, cv2.LINE_AA)
            # cv2.imwrite(fname, outimage)

            # self.framecnt += 1
            # if ((self.uccscart.tbdebuglevel>-1 )and self.framecnt < 3):
            #     self.debugstring += '  Writing '+ fname + 'with overlay'+ wstring
            #     print(self.debugstring)

    return action
