import numpy as np

import data_loader as data
import distance_calculator as dist_cal
import formatter as formatter
import probability as prob
import spatial_details as sp_details
from current_config import CurrentConfig


# get probability differene froom initial state


def istate_diff_EVT_prob(actual_state):
    # TODO: duplicate declaration, they are same. Move to a common place
    cur_conf = CurrentConfig()
    dimname = data.dimansion_name

    # load imin/imax from training..  with some extensions. From code some of these values don't seem plausable (blockx for example) but we saw them in training data.  maybe nic mixed up some parms/files but won't hurt too much fi we mis some
    # fitwblpy output for initial state data

    # no point in computing probabilities if we won't use them in scoring
    if cur_conf.episode > (cur_conf.scoreforKL * 2):
        return 0

    # max in min come directly for create_evm_data.py's output
    # if first time load up data..
    if cur_conf.imax is None:
        cur_conf.imax = data.imax
        cur_conf.imin = data.imin
        initwbl = data.initwbl

        cur_conf.ishape = initwbl[0, :, 0]
        cur_conf.iscale = initwbl[0, :, 2]

    imax = cur_conf.imax
    imin = cur_conf.imin
    ishape = cur_conf.ishape
    iscale = cur_conf.iscale

    initprob = 0  # assume nothing new in world

    cart_pos = [actual_state['cart']['x_position'], actual_state['cart']
    ['y_position'], actual_state['cart']['z_position']]
    cart_pos = np.asarray(cart_pos)

    charactermin = 1e-2

    istate = formatter.format_istate_data(actual_state)
    # do base state for cart(6)  and pole (7)   looking at position and velocity.
    for j in range(13):
        if abs(istate[j]) > imax[j]:
            probv = prob.awcdf(istate[j], imax[j], iscale[j], ishape[j])
            initprob += probv
            if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                cur_conf.logstr += "& P2+ LL1 " + "Step " + str(cur_conf.tick) + str(
                    dimname[j]) + " init increase  " + str(round(
                    istate[j], 3)) + " " + str(round(imax[j], 3)) + " " + str(round(probv, 3)) + " " + str(
                    round(iscale[j], 3))
                if not cur_conf.noveltyindicator:
                    cur_conf.logstr += "j=", str(j) + "state = " + str(cur_conf.current_state)

        if abs(istate[j]) < imin[j]:
            probv = prob.awcdf(
                abs(istate[j]), imin[j], iscale[j], ishape[j])
            if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                initprob += probv
                cur_conf.logstr += "& P2+ LL1" + "Step " + str(cur_conf.tick) + str(
                    dimname[j]) + " init decrease  " + str(
                    round(istate[j], 3)) + " " + str(round(imin[j], 3)) + " " + str(round(iscale[j], 3))
                if not cur_conf.noveltyindicator:
                    cur_conf.logstr += "j=", str(j) + \
                                       "state = " + str(cur_conf.current_state)

    # look for walls to be in bad position

    wallstart = len(istate) - 24
    if wallstart < 19:  # whould have at least 2 blocks
        probv = 1
        initprob += probv
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            cur_conf.logstr += "&P2+ LL2  " + "Step " + \
                               str(cur_conf.tick) + \
                               "  Block-quantity-decrease (Level L2 change) Len=" + \
                               wallstart
    if wallstart > 19 + 5 * 6:  # sould have at most 5 blocks
        probv = 1
        initprob += probv
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            cur_conf.logstr += "&P2+ LL2 " + "Step " + \
                               str(cur_conf.tick) + \
                               " Block-quantity-increase (Level L2 change) Len=" + \
                               wallstart

    # look for blocks to be in bad position or to have bad velocity

    k = 13  # where block data begins
    cur_conf.num_blocks = 0
    for j in range(13, wallstart, 1):
        probv = 0
        if abs(istate[j]) > imax[k]:
            probv = prob.awcdf(
                abs(istate[j]), imax[k], iscale[k], ishape[k])
            if (abs(istate[j]) - imax[k]) > 1:
                cur_conf.logstr += "& P2+ LL2  " + "Step " + str(cur_conf.tick) + str(
                    dimname[k]) + " init increase " + " " + str(round(istate[j], 3)) + " " + str(
                    round(imax[k], 3)) + " " + str(round(probv, 3)) + " " + str(round(iscale[j], 3)) + " " + str(
                    round(ishape[j], 3)) + " " + str(round(probv, 3))
                initprob += max(.24, probv)
            #                    self.logstr += "j="+ str(j)+ str(self.current_state)
            elif probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                cur_conf.logstr += "& P2+ LL2 " + "Step " + str(cur_conf.tick) + str(
                    dimname[k]) + " init increase " + " " + str(round(istate[j], 3)) + " " + str(
                    round(imax[k], 3)) + " " + str(round(probv, 3)) + " " + str(round(iscale[j], 3)) + " " + str(
                    round(ishape[j], 3)) + " " + str(round(probv, 3))
        #                    self.logstr += str(self.current_state)
        if abs(istate[j]) < imin[k]:
            if (imin[k] - abs(istate[j])) > 1:
                cur_conf.logstr += "& P2+ LL2 " + "Step " + str(cur_conf.tick) + str(
                    dimname[k]) + " init decrease " + " " + str(round(istate[j], 3)) + " " + str(
                    round(imax[k], 3)) + " " + str(round(probv, 3)) + " " + str(round(iscale[j], 3)) + " " + str(
                    round(ishape[j], 3)) + " " + str(round(probv, 3))
                #                    self.logstr += "j="+ str(j)+ str(self.current_state)
                initprob += max(.24, probv)
            else:
                probv = prob.awcdf(
                    abs(istate[j]), imin[k], iscale[k], ishape[k])
                initprob += probv
                if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                    cur_conf.logstr += "&P2+ LL2" + "Step " + str(cur_conf.tick) + " " + str(
                        dimname[k]) + " init decrease " + " " + str(round(istate[j], 3)) + " " + str(
                        round(imin[k], 3)) + " " + str(round(probv, 3)) + " " + str(round(iscale[j], 3)) + " " + str(
                        round(ishape[j], 3)) + " " + str(round(probv, 3))
        #                    self.logstr += "j="+ str(j)+ str(self.current_state)
        k = k + 1
        if (k == 19):
            cur_conf.num_blocks += 1
            k = 13  # reset for next block

    cur_conf.logstr += ";"

    # look for blocks to heading at cart's initital position

    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k
        #            dist = self.point_to_line_dist(istate[0:3],istate[k+nb*6:k+nb*6+2],istate[k+nb*6+3:k+nb*6+5])
        dist = dist_cal.point_to_line_dist(sp_details.cart_pos(istate),
                                           sp_details.block_pos(
                                               istate, nb),
                                           sp_details.block_vel(istate, nb))
        probv = 0
        if dist < 1e-3:  # should do wlb fit on this.. but for now just a hack
            probv = .5  # we should fix that
        elif dist < .01:  # should do wlb fit on this.. but for now just a hack
            probv = (.01 - dist) / (.01 - 1e-3)
            probv = .5 * probv * probv  # square it so its a bit more concentrated and smoother
        initprob += probv
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            cur_conf.logstr += "&P2+ LL3 " + "Step " + str(cur_conf.tick) + " P2+ Char Block " + str(
                nb) + " on initial direction attacking cart " + " with prob " + str(probv)
    #                self.logstr += str(self.current_state)

    # look for blocks motions that heading to  other blocks initital position
    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        for nb2 in range(nb + 1, cur_conf.num_blocks):
            #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k
            #                dist = self.point_to_line_dist(istate[k+nb2*6:k+nb2*6+2],istate[k+nb*6:k+nb*6+2],istate[k+nb*6+3:k+nb*6+5])
            dist = dist_cal.point_to_line_dist(sp_details.block_pos(istate, nb2),
                                               sp_details.block_pos(
                                                   istate, nb),
                                               sp_details.block_vel(istate, nb))
            probv = 0

            if dist < 1e-3:  # should do wlb fit on this.. but for now just a hack.  Note blocks frequently can randomly do this so don't consider it too much novelty.  Loose in test since they move before we see it
                probv = .4
            elif dist < .01:  # should do wlb fit on this.. but for now just a hack
                probv = .4 * (.01 - dist) / (.01)
            initprob += probv
            if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                cur_conf.logstr += "& P2+ LL5" + "Step " + str(cur_conf.tick) + " P2+ Char Block " + str(
                    nb) + " on initial direction aiming at block" + str(nb2) + " with prob " + str(probv)
    #                    self.logstr += str(self.current_state)

    # look for blocks motions that are parallel/or anti-parallel

    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        for nb2 in range(nb + 1, cur_conf.num_blocks):
            #  Use only block direction (velocity) for angle test  But if either vector norm is 0 then cannot compute angle, but in normal world block are never stationary so still abnormal
            angle = sp_details.vector_angle(sp_details.block_vel(
                istate, nb), sp_details.block_vel(istate, nb2))
            # get weibul probabilities for the angles..  cannot be both small and large and weibul go to zero fast enough we
            probv = 0
            if angle < .1:
                probv = prob.wcdf(angle, 0.00, .512, .1218)
            if angle > 3.1:
                probv = prob.rwcdf(angle, 3.14, .512, .1218)
            # since this can happon randomly we never let it take longer
            if probv > .5:
                probv = .5

            initprob += probv
            if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                cur_conf.logstr += "& P2+ LL3" + "Step " + str(cur_conf.tick) + " Char Block motion " + str(
                    nb) + " angle exception (e.g. parallel) to  block" + str(nb2) + " with prob " + str(
                    probv) + "for angle" + str(angle)
    #                    self.logstr += str(self.current_state)

    # look for blocks motions that are lines that will intersect

    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        for nb2 in range(nb + 1, cur_conf.num_blocks):
            dist = dist_cal.line_to_line_dist(sp_details.block_pos(istate, nb), sp_details.block_vel(
                istate, nb), sp_details.block_pos(istate, nb2), sp_details.block_vel(istate, nb2))
            # get weibul probabilities for the line-to-line-distance
            probv = 0
            if dist < .025:
                probw = prob.wcdf(angle, 0.013, .474, .136)
                # this can occur randomly so limit its impact
                probv = min(.2, probw)
                initprob += probv
                if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                    cur_conf.logstr += "& P2+ LL3" + "Step " + str(cur_conf.tick) + " P2+ Char Block  " + str(
                        nb) + " likely intersects with block" + str(
                        nb2) + " with probs " + str(probw) + " " + str(probv) + "for intersection distance" + str(dist)
    #                        self.logstr += str(self.current_state)

    cur_conf.logstr += ";"

    #        was limiting to one, but do want to allow more since  prob of novelty overall but we also discount this elesewhere
    if initprob > cur_conf.maxinitprob:
        cur_conf.logstr += "Iprob clamped from" + str(initprob)
        initprob = cur_conf.maxinitprob

    if initprob > cur_conf.minprob_consecutive:
        cur_conf.consecutiveinit = cur_conf.consecutiveinit + 1
        if cur_conf.consecutiveinit > 3:
            cur_conf.initprobscale = min(
                cur_conf.initprobscale + .25, cur_conf.maxinitprob)
        if cur_conf.uccscart.tbdebuglevel > 1:
            print("Initprob cnt char ", initprob, cur_conf.cnt, cur_conf.logstr)
    else:
        cur_conf.consecutiveinit = 0
    return initprob


# get probability differene from checking state difference from prediction


def cstate_diff_EVT_prob(cdiff, astate):
    # TODO: duplicate declaration, they are same. Move to a common place
    cur_conf = CurrentConfig()
    dimname = data.dimansion_name

    if cur_conf.episode > (cur_conf.scoreforKL * 3):
        return 0

    if cur_conf.dmax is None:
        # load data from triningn
        cur_conf.dmax = data.dmax
        diffwbl = data.diffwbl

        cur_conf.dshape = diffwbl[0, :, 0]
        cur_conf.dscale = diffwbl[0, :, 2]
    imax = cur_conf.dmax
    imin = -imax
    ishape = cur_conf.dshape
    iscale = cur_conf.dscale

    # else  just initize locals

    # need to be far enough along to get good prediciton
    if cur_conf.cnt < cur_conf.skipfirstNscores:
        return 0

    prob = 0  # where we accumualte probability
    probv = 0
    charactermin = 1e-2
    istate = cdiff
    # do base state for cart(6)  and pole (7) ..  because of noise we use only  a frac and ignore if around 1974 and if really large
    for j in range(13):
        if cur_conf.uccscart.tbdebuglevel > 0 and (istate[j] - imax[j]) >= 1.0:
            print("Step " + str(cur_conf.tick) + dimname[j] + " ignored diff increase with state/max " + str(
                round(istate[j], 5)) + " " + str(round(imax[j], 5)))
        if istate[j] > imax[j] and (istate[j] - imax[j]) < 1.0:
            #                probv =  prob.awcdf(istate[j],imax[j],iscale[j],ishape[j]);
            probv = prob.awcdf(
                abs(istate[j]), imax[j], iscale[j], ishape[j])
            # hack..    often the bug in system interface produces errors that have state around 1974.  Might skip a few real errors but should reduce false alarms a good bit
            if abs(abs(istate[j]) - .1974) < cur_conf.maxclampedprob / 2 and len(cur_conf.logstr) < cur_conf.maxcarlen:
                if cur_conf.uccscart.tbdebuglevel > 0:
                    cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + dimname[
                        j] + " 1974 clampingbad  prob " + " " + str(
                        round(probv, 5)) + "  large bad state " + str(round(istate[j], 5)) + " " + str(
                        round(imax[j], 5))
                probv = min(cur_conf.maxclampedprob, probv)
            else:
                if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                    cur_conf.logstr += "&" + "P2+ LL1 Step " + str(cur_conf.tick) + dimname[
                        j] + " diff increase prob " + " " + str(
                        round(probv, 5)) + " s/l " + str(round(istate[j], 5)) + " " + str(round(imax[j], 5))
        elif istate[j] < imin[j] and (imin[j] - istate[j]) < 1:
            if cur_conf.uccscart.tbdebuglevel > 0 and (imin[j] - istate[j]) >= 1:
                print("Step " + str(cur_conf.tick) + dimname[j] + " ignored diff too small with state/min " + str(
                    round(istate[j], 5)) + " " + str(round(imin[j], 5)))
            probv = prob.awcdf(abs(istate[j]), abs(
                imin[j]), iscale[j], ishape[j])
            # hack..    often the bug in system interface produces errors that have state around 1974.  Might skip a few real errors but should reduce false alarms a good bit
            if abs(abs(istate[j]) - .1974) < cur_conf.maxclampedprob / 2 and len(cur_conf.logstr) < cur_conf.maxcarlen:
                if cur_conf.uccscart.tbdebuglevel > 0:
                    cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + dimname[
                        j] + " 1974 clampingbad   prob " + " " + str(
                        round(probv, 5)) + "  small bad state " + str(round(istate[j], 5)) + " " + str(
                        round(imax[j], 5))
                probv = min(cur_conf.maxclampedprob, probv)
            else:
                if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                    cur_conf.logstr += "& P2+ LL1  Step " + str(cur_conf.tick) + dimname[
                        j] + " diff decrease prob " + " " + str(
                        round(probv, 5)) + "  s/l " + str(round(istate[j], 5)) + " " + str(round(imin[j], 5))

    prob += probv
    #        print("at Step ", self.tick, " Dyn state ",istate)

    #        if(self.episode > 20):        pdb.set_trace()

    probb = 0
    # no walls in dtate diff just looping over blocks
    k = 13  # for name max/ame indixing where we have only one block
    for j in range(13, len(istate), 1):
        # compute overall min/max for x,y position dimensions in actual state
        if (k == 13 or k == 14) and astate[j] > cur_conf.blockmax:
            cur_conf.blockmax = astate[j]
        if (k == 13 or k == 14) and astate[j] < cur_conf.blockmin:
            cur_conf.blockmin = astate[j]
        if k > 15 and k < 19:  # 16 17 and 18 are block velocity
            cur_conf.blockvelmax = max(astate[j], cur_conf.blockvelmax)

        # block motion not as predicted (domain independent test) but can be caused by may things and applies to L3, L5 and L7 as directions are off and  L4 since the bounce early produces a unepxcted position/velocity)
        # maybe some domain dependent stuff could differentiate
        # the random error (from block collisons I think) sometimes cause large errors, so have to treat this a s very noisey and limit impact and only apply when resonable
        if abs(istate[j]) > abs(imax[k]) and (abs(istate[j]) - (abs(imax[k]))) < .5:
            # some randome error stll creap in so limit is impact below
            probb += prob.awcdf(abs(istate[j]), abs(imax[k]), iscale[k], ishape[k])
            if probb > .001 and len(cur_conf.logstr) < cur_conf.maxcarlen:
                cur_conf.logstr += "&P2+ Block Motion Prediction Error" + "Step " + str(cur_conf.tick) + " " + str(
                    dimname[k]) + " diff increase, prob " + " " + str(round(probb, 5)) + "  s/l " + str(
                    round(istate[j], 5)) + " " + str(round(imax[k], 5))
        elif abs(istate[j]) > (abs(imax[k])):
            # some randome error stll creap in so limit is impact
            probb = prob.awcdf(abs(istate[j]), abs(imax[k]), iscale[k], ishape[k])
            if cur_conf.uccscart.tbdebuglevel > 0 and len(cur_conf.logstr) < cur_conf.maxcarlen:
                cur_conf.logstr += "&P2+ Block Motion Prediction Error " + "Step " + str(cur_conf.tick) + " " + str(
                    dimname[k]) + " diff way too large ignored prob " + " " + str(round(probb, 5)) + "  s/l " + str(
                    round(istate[j], 5)) + " " + str(round(imax[k], 5))

        k = k + 1
        if k == 19:
            k = 13  # reset for next block
    # if we want to not impact of limit l3/l7 errors which can happen rnadomly and there are many block and many steps
    cur_conf.dynblocksprob += min(cur_conf.maxdynamicprob, probb)

    # look for blocks motions that heading to  other blocks position
    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        for nb2 in range(nb + 1, cur_conf.num_blocks):
            #  cart in position 0,   blocks are  position and then velocity 3d vectors starting at location k
            probv = 0
            if len(sp_details.block_pos(astate, nb2)) == len(sp_details.block_pos(astate, nb)):
                dist = dist_cal.point_to_line_dist(sp_details.block_pos(astate, nb2),
                                                   sp_details.block_pos(
                                                       astate, nb),
                                                   sp_details.block_vel(astate, nb))

                if dist < 1e-3:  # should do wlb fit on this.. but for now just a hack.  Note blocks frequently can randomly do this so don't consider it too much novelty
                    probv = cur_conf.maxdynamicprob
                elif dist < .01:  # should do wlb fit on this.. but for now just a hack
                    probv = cur_conf.maxdynamicprob * (.01 - dist) / (.01)
                prob += probv
                if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                    cur_conf.logstr += "& P2+ LL5" + "Step " + str(cur_conf.tick) + " P2+ Char Block " + str(
                        nb) + " on diff  direction aiming at block" + str(nb2) + " with prob " + str(probv)
    #                    self.logstr += str(self.current_state)

    # look for blocks motions that are parallel/or anti-parallel

    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        for nb2 in range(nb + 1, cur_conf.num_blocks):
            #  Use only block direction (velocity) for angle test  But if either vector norm is 0 then cannot compute angle, but in normal world block are never stationary so still abnormal
            if len(sp_details.block_pos(astate, nb2)) == len(sp_details.block_pos(astate, nb)):
                angle = sp_details.vector_angle(sp_details.block_vel(
                    astate, nb), sp_details.block_vel(astate, nb2))
                # get weibul probabilities for the angles..  cannot be both small and large and weibul go to zero fast enough we
                deltav = 0
                if angle < .02:
                    deltav = .01 + \
                             prob.wcdf(angle, 0.00, .512, .1218)
                if angle > 3.12:
                    deltav = .01 + \
                             prob.rwcdf(angle, 3.14, .512, .1218)
                if deltav > 0 and len(cur_conf.logstr) < cur_conf.maxcarlen:
                    cur_conf.logstr += "& P2+ LL3 or LL5" + "Step " + str(cur_conf.tick) + " Char Block motion " + str(
                        nb) + " dyn Block-Toward-Block " + str(
                        nb2) + " with probs " + str(deltav) + " " + str(probv) + " for angle " + str(round(angle, 3))
                probv += deltav
    # since this can happon randomly we never let it take longer
    if probv > cur_conf.maxdynamicprob:
        probv = cur_conf.maxdynamicprob
    prob += min(probv, cur_conf.maxdynamicprob)

    # look for blocks motions that are lines that will intersect
    probv = 0
    k = 13  # where block data begins
    for nb in range(cur_conf.num_blocks):
        for nb2 in range(nb + 1, cur_conf.num_blocks):
            if len(sp_details.block_pos(astate, nb2)) == len(sp_details.block_pos(astate, nb)):
                dist = dist_cal.line_to_line_dist(sp_details.block_pos(astate, nb), sp_details.block_vel(
                    astate, nb), sp_details.block_pos(astate, nb2), sp_details.block_vel(astate, nb2))
                # get weibul probabilities for the line-to-line-distance
                probv = 0
                if dist < .025:
                    probw = .01 + prob.wcdf(angle, 0.013, .474, .136)
                    probv += probw  # this can occur randomly so limit its impact
                    if probw > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
                        cur_conf.logstr += "& P2+ LL3 or LL5" + "Step " + str(
                            cur_conf.tick) + " P2+ Char Block  " + str(nb) + " dyn likely intersects with block" + str(
                            nb2) + " with probs " + str(probw) + " " + str(probv) + "for intersection distance" + str(
                            dist)

    prob += min(probv, cur_conf.maxdynamicprob)
    cur_conf.logstr += ";"

    prob = min(cur_conf.maxdynamicprob, prob)

    if prob > 1:
        prob = 1

    return prob

    # get probability differene froom initial state


# Finding no use case
def istate_diff_G_prob(actual_state):
    dimname = data.dimansion_name

    # load mean/std from training..
    # if first time load up data..

    cur_conf = CurrentConfig()
    if cur_conf.imean is None:
        # abuse tha terms since we are tryng to share code with  EVT version
        cur_conf.imean = data.imean
        # adjusted based on code..
        cur_conf.istd = data.istd

    imean = cur_conf.imean
    istd = cur_conf.istd

    initprob = 0  # assume nothing new in world

    cart_pos = [actual_state['cart']['x_position'], actual_state['cart']
    ['y_position'], actual_state['cart']['z_position']]
    cart_pos = np.asarray(cart_pos)

    charactermin = 1e-2

    istate = formatter.format_istate_data(actual_state)
    # do base state for cart(6)  and pole (7)
    for j in range(13):
        probv = cur_conf.gcdf(istate[j], imean[j], istd[j])
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            #                initprob += probv
            initprob = max(initprob, probv)
            cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + str(dimname[j]) + " init out of range  " + str(
                round(
                    istate[j], 3)) + " " + str(round(imean[j], 3)) + " " + str(round(istd[j], 3)) + " " + str(
                round(probv, 3))

    wallstart = len(istate) - 24
    k = 13  # for name max/ame indixing where we have only one block
    for j in range(13, wallstart, 1):
        if "Wall" in str(dimname[j]):
            break
        probv = cur_conf.gcdf(istate[j], imean[j], istd[j])
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            #                initprob += probv
            initprob = max(initprob, probv)
            cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + str(dimname[j]) + "  init out of range  " + str(
                round(
                    istate[j], 3)) + " " + str(round(imean[j], 3)) + " " + str(round(istd[j], 3)) + " " + str(
                round(probv, 3))
        k = k + 1
        if k == 19:
            k = 13  # reset for next block
    cur_conf.logstr += ";"

    if initprob > 1:
        initprob = 1

    if initprob > cur_conf.minprob_consecutive:
        cur_conf.consecutiveinit = min(
            cur_conf.consecutiveinit + 1, cur_conf.maxconsecutivefailthresh)
        if cur_conf.uccscart.tbdebuglevel > 1:
            print("Initprob cnt char ", initprob, cur_conf.cnt, cur_conf.logstr)
    else:
        cur_conf.consecutiveinit = 0
    return initprob


# get probability differene froom continuing state difference


def cstate_diff_G_prob(cdiff):
    dimname = data.dimansion_name

    cur_conf = CurrentConfig()
    if cur_conf.dmean is None:
        # load data from triningn
        cur_conf.dmean = data.dmean
        cur_conf.dstd = data.dstd

    imean = cur_conf.dmean
    istd = cur_conf.dstd

    if cur_conf.cnt < cur_conf.skipfirstNscores:
        return 0  # need to be far enough along to get good prediciton

    prob = 0  # where we accumualte probability
    charactermin = 1e-2
    istate = cdiff
    # do base state for cart(6)  and pole (7)
    for j in range(13):
        probv = cur_conf.gcdf(istate[j], imean[j], istd[j])
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            #                prob += probv
            prob = max(prob, probv)
            cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + str(dimname[j]) + " dyn out of range  " + str(round(
                istate[j], 6)) + " " + str(round(imean[j], 6)) + " " + str(round(istd[j], 6)) + " " + str(
                round(probv, 3))

    # no walls in dtate diff just looping over blocks

    k = 13  # for name max/ame indixing where we have only one block
    for j in range(13, len(istate), 1):
        if "Wall" in str(dimname[j]):
            break
        probv = cur_conf.gcdf(istate[j], imean[j], istd[j])
        if probv > charactermin and len(cur_conf.logstr) < cur_conf.maxcarlen:
            prob += probv
            prob = max(prob, probv)
            cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + str(dimname[j]) + " dyn out of range  " + str(round(
                istate[j], 6)) + " " + str(round(imean[j], 6)) + " " + str(round(istd[j], 6)) + " " + str(
                round(probv, 3))
        k = k + 1
        if k == 19:
            k = 13  # reset for next block
    cur_conf.logstr += ";"

    if prob > cur_conf.minprob_consecutive:
        if prob > 1:
            prob = 1

    return prob
