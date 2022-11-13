import math

import numpy as np

import data_loader as data
from current_config import CurrentConfig


#####!!!!!##### Start INDEPNDENT CODE for EVT-
def kullback_leibler(self, mu, sigma, m, s):
    """
        Compute Kullback Leibler with Gaussian assumption of training data
        mu: mean of test batch
        sigm: standard deviation of test batch
        m: mean of all data in training data set
        s: standard deviation of all data in training data set
        return: KL ditance, non negative double precison float
    """

    sigma = max(sigma, .0000001)
    s = max(s, .0000001)
    kl = np.log(s / sigma) + (((sigma ** 2) + ((mu - m) ** 2)) / (2 * (s ** 2))) - 0.5
    return kl


# reversed wbl for maxim fitting
def rwcdf(self, x, iloc, ishape, iscale):
    if x - iloc < 0:
        prob = 0
    else:
        prob = 1 - math.pow(math.exp(-(x - iloc) / iscale), ishape)
    # if(prob > 1e-4): print("in rwcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob

    # abs wbl for unsided fitting


def awcdf(self, x, iloc, ishape, iscale):
    prob = 1 - math.pow(math.exp(-abs(x - iloc) / iscale), ishape)
    # if(prob > 1e-4): print("in awcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob


# regualr wbl for minimum fits
def wcdf(self, x, iloc, ishape, iscale):
    if iloc - x < 0:
        prob = 0
    else:
        prob = 1 - math.pow(math.exp(-(iloc - x) / iscale), ishape)
    # if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob


#####!!!!!##### End Doimain Independent CODE for EVT-


def world_change_prob(self, settrain=False):
    cur_conf = CurrentConfig()
    # don't let first episodes  impact world change.. need stabilsied scores/probabilites.. skipping work here also
    # makes it faster
    if cur_conf.episode < cur_conf.scoreforKL:
        cur_conf.worldchangedacc = 0
        cur_conf.worldchangeblend = 0
        cur_conf.previous_wc = 0
        return cur_conf.worldchangedacc

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "worldchanged went down, line 1193" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.logstr += "&" + cur_conf.debugstring
        cur_conf.worldchangedacc = cur_conf.previous_wc

    cur_conf.summary = ""  # reset summary to blank

    mlength = len(cur_conf.problist)
    mlength = min(cur_conf.scoreforKL, mlength)
    # we look at the larger of the begging or end of list.. world changes most obvious at the ends.

    window_width = 11
    # look at list of performacne to see if its deviation from training is so that is.. skip more since it needs to
    # be stable for window smoothing+ mean/variance computaiton
    PerfKL = 0
    if len(cur_conf.perflist) > (cur_conf.scoreforKL + window_width) and len(
            cur_conf.perflist) < 3 * cur_conf.scoreforKL:
        # get smoothed performance
        cumsum_vec = np.cumsum(np.insert(cur_conf.perflist, 0, 0))
        smoothed = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        pmu = np.mean(smoothed[:-cur_conf.scoreforKL])  # we skip first/iniiprob... it is used elsehwere.
        psigma = np.std(smoothed[:-cur_conf.scoreforKL])

        # if(pmu <  self.mean_perf or pmu >  self.mean_perf +  self.stdev_perf):     #if we want only  KL for those
        # what have worse performance or much better
        if pmu > cur_conf.mean_perf:  # if we want only  KL for those what have worse performance or much better
            PerfKL = cur_conf.kullback_leibler(pmu, psigma, cur_conf.mean_perf, cur_conf.stdev_perf)
            cur_conf.debugstring = '   PerfKL {} {} {} {} ={} ,'.format(pmu, psigma, cur_conf.mean_perf,
                                                                        cur_conf.stdev_perf,
                                                                        round(PerfKL, 3))
            print(cur_conf.debugstring)

        # If there is still too much variation (too many FP) in the variance in the small window so we use stdev and
        # just new mean this allows smaller (faster) window for detection. PerfKL = self.kullback_leibler(pmu,
        # self.stdev_perf, self.mean_perf, self.stdev_perf)
    else:
        PerfKL = 0

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "worldchanged went down, line 1222" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.logstr += "&" + cur_conf.debugstring
        cur_conf.worldchangedacc = cur_conf.previous_wc

    if mlength > 1:
        mu = np.mean(cur_conf.problist[0:mlength - 1])
        sigma = np.std(cur_conf.problist[0:mlength - 1])
    else:
        mu = sigma = 0
        cur_conf.debugstring = '***Zero Lenth World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} val {} ' \
                               'thresh {} {} scores{}'.format(
            round(cur_conf.worldchangedacc, 5), [round(num, 2) for num in cur_conf.problist], round(mu, 3),
            round(sigma, 3),
            round(cur_conf.mean_train, 3), round(cur_conf.stdev_train, 3), round(cur_conf.KL_val, 3),
            round(cur_conf.KL_threshold, 3),
            "\n", [round(num, 2) for num in cur_conf.scorelist])
        print(cur_conf.debugstring)
        cur_conf.logstr += "&" + cur_conf.debugstring
        cur_conf.worldchanged = cur_conf.worldchangedacc
        return max(cur_conf.worldchangedacc, cur_conf.previous_wc)

    if settrain:
        cur_conf.mean_train = mu
        cur_conf.stdev_train = sigma
        print("Set  world change train mu and sigma", mu, sigma)
        cur_conf.logstr += "&" + "Set  world change train mu and sigma" + str(mu) + str(
            sigma) + " saying world_change = 0"
        cur_conf.worldchanged = 0
        return max(cur_conf.worldchangedacc, cur_conf.previous_wc)

    if cur_conf.mean_train == 0:
        cur_conf.mean_train = 0.004
        cur_conf.stdev_train = 0.009
        cur_conf.dynam_prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.

    if mu > cur_conf.mean_train:
        cur_conf.KL_val = cur_conf.kullback_leibler(mu, sigma, cur_conf.mean_train, cur_conf.stdev_train)
    else:
        cur_conf.KL_val = 0
    cur_conf.debugstring = '***Short World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh ' \
                           '{} {} scores{}'.format(
        round(cur_conf.worldchangedacc, 3), [round(num, 2) for num in cur_conf.problist], round(mu, 3), round(sigma, 3),
        round(cur_conf.mean_train, 3),
        round(cur_conf.stdev_train, 3), round(cur_conf.KL_val, 5), round(cur_conf.KL_threshold, 5), "\n",
        [round(num, 2) for num in cur_conf.scorelist])
    if cur_conf.debug:
        print(cur_conf.debugstring)

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "   worldchanged went down, line 1260" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.logstr += "&" + cur_conf.debugstring
        cur_conf.worldchangedacc = cur_conf.previous_wc

    dprob = perfprob = 0  # don't allow on short runs.. dynamics and performance are off

    if len(cur_conf.problist) < 198:  # for real work but short list
        cur_conf.consecutivesuccess = 0
        cur_conf.failcnt += 1
        if cur_conf.consecutivefail > 0:
            cur_conf.consecutivefail = min(cur_conf.consecutivefail + 1, cur_conf.maxconsecutivefailthresh + 2)
            if cur_conf.consecutivefail > cur_conf.maxconsecutivefail:
                cur_conf.maxconsecutivefail = cur_conf.consecutivefail
                if cur_conf.maxconsecutivefail > cur_conf.maxconsecutivefailthresh:
                    cur_conf.worldchangedacc = 1
                    cur_conf.logstr += "&" + "Step " + str(
                        cur_conf.tick) + "#####? Uncontrollable world -- too many consecutive failures.  Guessing " \
                                         "actions were remapped/perturbed but will take a while to confirm ##### "

        else:
            cur_conf.consecutivefail = 1
        if mu > cur_conf.mean_train:
            cur_conf.KL_val = cur_conf.kullback_leibler(mu, sigma, cur_conf.mean_train, cur_conf.stdev_train)
        else:
            cur_conf.KL_val = 0

        cur_conf.debugstring = '***Short World Change Acc={}, Failcnt= {} Prob ={},mu={}, sigmas {}, mean {} stdev{} ' \
                               'KLval {} thresh {} {}        scores{}'.format(
            round(cur_conf.worldchangedacc, 3), cur_conf.failcnt, [round(num, 2) for num in cur_conf.problist],
            round(mu, 3),
            round(sigma, 3), round(cur_conf.mean_train, 3),
            round(cur_conf.stdev_train, 3), round(cur_conf.KL_val, 5), round(cur_conf.KL_threshold, 5), "\n",
            [round(num, 2) for num in cur_conf.scorelist])
        if cur_conf.debug:
            print(cur_conf.debugstring)

    else:
        cur_conf.consecutivefail = 0
        cur_conf.consecutivesuccess += 1
        if cur_conf.consecutivesuccess > cur_conf.maxconsecutivesuccess:
            cur_conf.maxconsecutivesuccess = cur_conf.consecutivesuccess

        if sigma == 0:
            if mu == cur_conf.mean_train:
                cur_conf.debugstring = 'BadSigma World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} ' \
                                       'KLval {} thresh {} {} scores{}'.format(
                    round(cur_conf.worldchangedacc, 3), [round(num, 2) for num in cur_conf.problist], round(mu, 3),
                    round(sigma, 3), round(cur_conf.mean_train, 3),
                    round(cur_conf.stdev_train, 3), round(cur_conf.KL_val, 3), round(cur_conf.KL_threshold, 3), "\n",
                    [round(num, 2) for num in cur_conf.scorelist])
                print(cur_conf.debugstring)
                cur_conf.logstr += "&" + cur_conf.debugstring
                return max(cur_conf.worldchangedacc, cur_conf.previous_wc)
        else:
            sigma = cur_conf.stdev_train

        if mu < cur_conf.mean_train:  # no point computing if world differences are smaller, it may be "much" smaller
            # but that is okay
            cur_conf.KL_val = 0
        else:
            cur_conf.KL_val = cur_conf.kullback_leibler(mu, sigma, cur_conf.mean_train, cur_conf.stdev_train)

        # KLscale = (self.num_epochs + 1 - self.episode / 2) / self.num_epochs  # decrease scale (increase
        # sensitvity)  from start 1 down to  1/2 KLscale = min(1, 4*(1 + self.episode) / num_epochs)  # decrease
        # scale (increase sensitvity)  from start 1 down to  1/2
        KLscale = 1
        dprob = min(1.0, (KLscale * cur_conf.KL_val))
        perfprob = min(1.0,
                       cur_conf.PerfScale * PerfKL)  # make this smaller since it is slowly varying and  added every
        # time.. less sensitive (good for FP avoid, sloer l

    # if we had  collisons and not consecuretive faliures, we don't use this episode for dynamic probability ..
    # collisions are not well predicted tlen = min(self.scoreforKL,len(self.uccscart.char))

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "   worldchanged went down, line 1325" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.logstr += "&" + cur_conf.debugstring
        cur_conf.worldchangedacc = cur_conf.previous_wc

    # random collisions can occur and they destroy probability computation so ignore them
    if (cur_conf.logstr.count("CP") > 4) and ("attack" not in cur_conf.logstr) and (cur_conf.consecutivefail < 4):
        prob = 0
        dprob = .0009990
        perfprob = 0
        print("Debug found ", cur_conf.logstr.count("CP"), "CPs in string")
    else:
        prob = min(1, max(dprob, perfprob))  # use max of dynamic and long-term performance probabilities.
        print("Debug did not find many CP ", cur_conf.logstr.count("CP"), " in string. Prob=", str(prob))

        # infrequent checkto outputting cnts for setinng up wbls for actual cnts to use to see if we whould update
        # world change
    if ((cur_conf.episode - 10) == cur_conf.scoreforKL) or ((cur_conf.episode - 10) == 2 * cur_conf.scoreforKL):
        cntval = np.zeros(15)
        cntprob = np.zeros(21)
        i = 0
        scale = 1. / int(cur_conf.episode)
        cntval[i] = initcnt = scale * cur_conf.trialchar.count("init")
        i += 1
        cntval[i] = blockcnt = scale * cur_conf.trialchar.count("Block")
        i += 1
        cntval[i] = blockvelcnt = scale * cur_conf.trialchar.count("Block Vel")
        i += 1
        cntval[i] = polecnt = scale * cur_conf.trialchar.count("Pole")
        i += 1
        cntval[i] = cartcnt = scale * cur_conf.trialchar.count("Cart")
        i += 1
        cntval[i] = smallcnt = scale * cur_conf.trialchar.count("Cart")
        i += 1
        cntval[i] = largecnt = scale * cur_conf.trialchar.count("Cart")
        i += 1
        cntval[i] = diffcnt = scale * cur_conf.trialchar.count("diff")
        i += 1
        cntval[i] = velcnt = scale * cur_conf.trialchar.count("Vel")
        i += 1
        cntval[i] = failcnt = scale * cur_conf.trialchar.count("High")
        i += 1
        cntval[i] = attcart = scale * cur_conf.trialchar.count("attacking cart")
        i += 1
        cntval[i] = aimblock = scale * cur_conf.trialchar.count("aiming")
        i += 1
        cntval[i] = parallelblock = scale * cur_conf.trialchar.count("parallel")
        levelcnt = np.zeros(9)
        i = 0
        i += 1
        levelcnt[i] = L1 = scale * cur_conf.trialchar.count("LL1")
        i += 1
        levelcnt[i] = L2block = scale * cur_conf.trialchar.count("LL2")
        i += 1
        levelcnt[i] = L3block = scale * cur_conf.trialchar.count("LL3")
        i += 1
        levelcnt[i] = L4block = scale * cur_conf.trialchar.count("LL4")
        i += 1
        levelcnt[i] = L5block = scale * cur_conf.trialchar.count("LL5")
        i += 1
        levelcnt[i] = L6block = scale * cur_conf.trialchar.count("LL6")
        i += 1
        levelcnt[i] = L7block = scale * cur_conf.trialchar.count("LL7")
        i += 1
        levelcnt[i] = L8block = scale * cur_conf.trialchar.count("LL8")

        """
            New String matching for json output 
            ncval = np.zeros("30")
            i+= 1; cntval[i]= self.trialchar.count("Block-Toward-Block")
    
            Block-quantity-increase
            Block-quantity-decrease
            Cart
            Cart Vel
            Pole
            Pole Vel
            Block
            Block Vel
            Wall
        """

        cntwbl = data.cntwbl
        cntmax = 0
        for i in range(12):
            cntprob[i] = cur_conf.wcdf(-cntval[i], cntwbl[i, 1], cntwbl[i, 0], cntwbl[i, 2])
            cntmax = max(cntmax, cntprob[i])

        if cntmax > .1: cntmax = .1  # limit impact this this is really a cumulative test on things we have
        # already seen

        # approximate wibul for  count of SA
        cntprob[13] = cur_conf.wcdf(152.73 - cntval[13], 0.1843, 2.01214, 36.5311)
        cntmax += cntprob[13]

        if prob < cntmax:
            cur_conf.logstr += 'Using detect as prob. detectcnts: {} dectprob {} '.format(cntval, cntprob)
            prob = max(prob, cntmax)
        else:
            cur_conf.logstr += 'detectcnts: {} dectprob {} '.format(cntval, cntprob)

    if len(cur_conf.problist) < cur_conf.scoreforKL:
        cur_conf.worldchanged = prob * len(cur_conf.problist) / (cur_conf.scoreforKL)
    elif len(cur_conf.problist) < 2 * cur_conf.scoreforKL:
        cur_conf.worldchanged = prob
    else:  # if very long list, KL beceomes too long, its more likely to be higher from random agent crashing into
        # pole so we the impact
        cur_conf.worldchanged = prob * (2 * cur_conf.scoreforKL) / len(cur_conf.problist)

    # only do blockmin/max if we did a long enough  note this does not need any blending or scoreforKL since its an
    # absolute novelty to have this happen if the blockmax/min (in general from past episode) from past are not
    # normal add them to the worldchange acce.  Again should be EVT based but not enough training yet.
    if cur_conf.episode > (
            cur_conf.scoreforKL) and cur_conf.tick > 170 and cur_conf.blockvelmax > 5:  # if we have been through
        # enough episodes and enough steps and blocks moved enough
        minmaxupdate = abs(cur_conf.blockmin - cur_conf.normblockmin) + abs(cur_conf.blockmax - cur_conf.normblockmax)
        if minmaxupdate > .2:
            if cur_conf.blockmin < cur_conf.normblockmin or cur_conf.blockmax > cur_conf.normblockmax:
                cur_conf.logstr += '&& P2+ LL4 BLock Size decreased. minmaxupdate=' + str(minmaxupdate)
            elif cur_conf.blockmin > cur_conf.normblockmin or cur_conf.blockmax < cur_conf.normblockmax:
                cur_conf.logstr += '&& P2+ LL4 BLock Size increased ' + str(minmaxupdate)
            print('  minmaxupdate {} min {} max {} normal {} {}  ,'.format(minmaxupdate, cur_conf.blockmin,
                                                                           cur_conf.blockmax,
                                                                           cur_conf.normblockmin,
                                                                           cur_conf.normblockmax))
            prob = prob + .3
            cur_conf.worldchanged += .3
            cur_conf.debugstring = 'Prob {} after Level LL4 Size Change minmaxupdate {} min {} max {} normal {} {} VelMax ' \
                                   '{},'.format(
                prob, minmaxupdate, cur_conf.blockmin, cur_conf.blockmax, cur_conf.normblockmin, cur_conf.normblockmax,
                cur_conf.blockvelmax)
            print(cur_conf.debugstring)
            cur_conf.logstr += "&" + cur_conf.debugstring

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "worldchanged went down, line 1314" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.logstr += "&" + cur_conf.debugstring
        cur_conf.worldchangedacc = cur_conf.previous_wc

    outputstats = False

    #####!!!!!##### end GLue CODE for EVT

    #####!!!!!#####  Domain Independent code tor consecurtiv efailures
    failinc = 0
    # if we are beyond KL window all we do is watch for failures to decide if we world is changed

    if cur_conf.scoreforKL + 1 < cur_conf.episode < 3 * cur_conf.scoreforKL:
        faildiff = cur_conf.failcnt / (cur_conf.episode + 1) - cur_conf.failfrac
        if faildiff > 0:
            cur_conf.logstr += "&" + "Step " + str(cur_conf.tick) + "High FailFrac=" + str(
                cur_conf.failcnt / (cur_conf.episode + 1))
            failinc = max(0, ((faildiff) * cur_conf.failscale))
            failinc *= min(1, (
                    cur_conf.episode - cur_conf.scoreforKL) / cur_conf.scoreforKL)  # Ramp it up slowly as its more unstable when it first starts at scoreforKL
            failinc = min(1, failinc)

        # world change blend  can go up or down depending on how probablites vary.. goes does allows us to ignore
        # spikes from uncommon events. as the bump i tup but eventually go down.
        if prob < .5 and self.worldchangedacc < .5:  # blend wo  i.e. decrease world change accumulator to limit
            # impact of randome events self.worldchangeblend = min(self.worldchangedacc * cur_conf.blenddownrate,
            # (self.blenddownrate *self.worldchanged + (1-self.blenddownrate) * self.worldchangeblend ))
            cur_conf.worldchangedacc = min(1, cur_conf.worldchangedacc * cur_conf.worldchangeblend)
            cur_conf.debugstring = "BlendDown "

        else:
            # worldchange acc once its above .5 it cannot not go down.. it includes max of old value..
            cur_conf.worldchangeblend = min(1, (
                    cur_conf.blenduprate * cur_conf.worldchanged + (
                    1 - cur_conf.blenduprate) * cur_conf.worldchangeblend))
            cur_conf.debugstring = "Blendup using rate " + str(cur_conf.blenduprate) + "wc/wcb=" + str(
                cur_conf.worldchanged) + " " + str(cur_conf.worldchangeblend)
            # we add in an impusle each step if the first step had initial world change.. so that accumulates over time

            if len(cur_conf.problist) > 0:
                cur_conf.worldchangedacc = min(1, cur_conf.problist[0] * cur_conf.initprobscale + (
                        cur_conf.worldchangedacc + cur_conf.worldchanged) * min(1, cur_conf.worldchangeblend + failinc))
            else:
                cur_conf.worldchangedacc = min(1, max(cur_conf.worldchangedacc + cur_conf.worldchanged,
                                                      cur_conf.worldchangedacc * cur_conf.worldchangeblend + failinc))
            cur_conf.debugstring += '    mu={}, sig {}, mean {} stdev{}  WCacc WBlend {} {} vals {} {} thresh {} '.format(
                round(mu, 3), round(sigma, 3), round(cur_conf.mean_train, 3), round(cur_conf.stdev_train, 3),
                round(cur_conf.worldchangedacc, 3), round(cur_conf.worldchangeblend, 3), round(cur_conf.KL_val, 3),
                round(PerfKL, 3), "\n")
            print(cur_conf.debugstring)

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "   worldchanged when down, line 1358" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.worldchangedacc = cur_conf.previous_wc

        # Until we have enough data, reset and worldchange, don't start accumulating
    if cur_conf.episode < cur_conf.scoreforKL:
        cur_conf.worldchangedacc = 0
        cur_conf.worldchangeblend = 0

        # normal world gets randome incremenets not bigger ones all in a row..  novel world can get many
        # consecutitive one so we increase prob of that big an increment and expodentially based on sequence length
        # and . Parms weak as not much data in training as its too infrequently used.  Not used later in stage as
        # random errors seem to grow with resets so consecurive random becomes more likely
    if (
            cur_conf.worldchangedacc - cur_conf.previous_wc) > cur_conf.minprob_consecutive and cur_conf.episode < 2 * cur_conf.scoreforKL:
        if cur_conf.consecutivewc > 0:
            wcconsecutivegrowth = (cur_conf.consecutivewc / 20) * (
                    1 - cur_conf.wcdf((cur_conf.worldchangedacc - cur_conf.previous_wc), .001, .502, .13))
            print("wcconsecutivegrowth = ", wcconsecutivegrowth, cur_conf.worldchangedacc, cur_conf.previous_wc)
            cur_conf.logstr += 'World Change Consecutive {}, prev {} base {} increent {} '.format(
                cur_conf.consecutivewc,
                round(cur_conf.previous_wc,
                      3), round(
                    cur_conf.worldchangedacc, 3), round(wcconsecutivegrowth, 3))
            cur_conf.worldchangedacc = min(1, cur_conf.worldchangedacc + wcconsecutivegrowth)
            print("EPi2 previs new world change", cur_conf.episode, cur_conf.previous_wc, cur_conf.worldchangedacc)

        cur_conf.consecutivewc += 1
    else:
        cur_conf.consecutivewc = 0
    #####!!!!!#####  End Domain Independent code tor consecurtiv efailures

    #####!!!!!#####  Start API code tor reporting
    cur_conf.logstr += 'World Change Acc={} {} {} {}, CW={},CD={} D/KL Probs={},{}'.format(
        round(cur_conf.worldchangedacc, 3),
        round(cur_conf.worldchangeblend, 3),
        round(cur_conf.previous_wc, 3),
        round(failinc, 3),
        cur_conf.consecutivewc,
        cur_conf.dynamiccount,
        round(dprob, 3),
        round(perfprob, 3))

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "   worldchanged went down, line 1389" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)

    print("EPi previs new world change", cur_conf.episode, cur_conf.previous_wc, cur_conf.worldchangedacc)
    if cur_conf.previous_wc < .5 and cur_conf.worldchangedacc >= .5:
        if cur_conf.noveltyindicator:
            cur_conf.logstr += cur_conf.hint
            cur_conf.logstr += "#!#!#!  World change TP Detection " + str(cur_conf.episode) + "  @@ FV= "
            cur_conf.summary += "#!#!#!  World change TP Detection " + str(cur_conf.episode) + "  @@  "
        elif not cur_conf.noveltyindicator:
            cur_conf.logstr += cur_conf.hint
            cur_conf.logstr += "#!#!#!  World change FP Detection " + str(cur_conf.episode) + "  @@ FV= "
            cur_conf.summary += "#!#!#!  World change FP Detection " + str(cur_conf.episode) + "  @@  "
        else:
            cur_conf.logstr += "#!#!#!  World change blind Detection " + str(cur_conf.episode) + "  @@ FV= "
        cur_conf.logstr += str(cur_conf.current_state)
        print("   Detecting at Episode==", cur_conf.episode, " hint=", cur_conf.hint)
        outputstats = True

        # if world changed an dour performance is below .65  we start using avoidance reaction
    #        if(self.worldchangedacc        >= .6 and (100*self.perf/self.totalcnt) < 65) :
    #                self.uccscart.use_avoid_reaction=True

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "   worldchanged went down, line 1403" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.worldchangedacc = cur_conf.previous_wc

    outputstats = True
    finalepisode = False
    if (cur_conf.episode + 1) % 99 == 0: finalepisode = True
    if finalepisode or outputstats:
        initcnt = cur_conf.trialchar.count("init")
        blockcnt = cur_conf.trialchar.count("Block")
        blockvelcnt = cur_conf.trialchar.count("Block Vel")
        blockmotion = cur_conf.trialchar.count("Block Motion")
        polecnt = cur_conf.trialchar.count("Pole")
        cartcnt = cur_conf.trialchar.count("Cart")
        smallcnt = cur_conf.trialchar.count("small")
        inccnt = cur_conf.trialchar.count("increase")
        deccnt = cur_conf.trialchar.count("decrease")
        largecnt = cur_conf.trialchar.count("large")
        diffcnt = cur_conf.trialchar.count("diff")
        velcnt = cur_conf.trialchar.count("Vel")
        failcnt = cur_conf.trialchar.count("High")
        attcart = cur_conf.trialchar.count("Relational Attacking cart")
        aimblock = cur_conf.trialchar.count("Relational aiming")
        parallelblock = cur_conf.trialchar.count("Relational parallel")
        levelcnt = np.zeros(10)
        i = 0
        i += 1
        levelcnt[i] = L1 = cur_conf.trialchar.count("LL1")
        i += 1
        levelcnt[i] = L2block = cur_conf.trialchar.count("LL2")
        i += 1
        levelcnt[i] = L3block = cur_conf.trialchar.count("LL3")
        i += 1
        levelcnt[i] = L4block = cur_conf.trialchar.count("LL4")
        i += 1
        levelcnt[i] = L5block = cur_conf.trialchar.count("LL5")
        i += 1
        levelcnt[i] = L6block = cur_conf.trialchar.count("LL6")
        i += 1
        levelcnt[i] = L7block = cur_conf.trialchar.count("LL7")
        i += 1
        levelcnt[i] = L8block = cur_conf.trialchar.count("LL8")

        # if we get enough L5 we increase our world change estimate
        if L5block > 50 and cur_conf.episode < 2.5 * cur_conf.scoreforKL:
            cur_conf.worldchangedacc = min(1, .25 + cur_conf.worldchangedacc)

        maxi = np.argmax(levelcnt)

        # if level is 8 is already filled in,  no need to do any filling
        if cur_conf.uccscart.characterization['level'] == int(8) or maxi == 8:
            # 8 had special code to do increase vs increasing
            cur_conf.uccscart.characterization['level'] = int(8)
            cur_conf.uccscart.characterization['entity'] = "Block"
            cur_conf.uccscart.characterization['attribute'] = "quantity"
            if (cur_conf.logstr.count("LL8: Blocks quantity dec") > cur_conf.logstr.count(
                    "LL8: Blocks quantity inc")):  # if we had more than one chance its increasing
                if (cur_conf.logstr.count(
                        "LL8: Blocks quantity dec") > 2):  # if we had more than one chance its increasing
                    cur_conf.uccscart.characterization['change'] = 'decreasing'
                else:
                    cur_conf.uccscart.characterization['change'] = 'decrease'
            else:
                if (cur_conf.logstr.count(
                        "LL8: Blocks quantity inc") > 2):  # if we had more than one chance its increasing
                    cur_conf.uccscart.characterization['change'] = 'increasing'
                else:
                    cur_conf.uccscart.characterization['change'] = 'increase'
        else:
            cur_conf.uccscart.characterization['level'] = int(maxi)
            cur_conf.uccscart.characterization['level'] = None
            cur_conf.uccscart.characterization['entity'] = None
            cur_conf.uccscart.characterization['attribute'] = None
            cur_conf.uccscart.characterization['change'] = None

            if maxi == 1 and levelcnt[1] > 1000:
                cur_conf.uccscart.characterization['level'] = int(1)
                if cartcnt > polecnt:
                    cur_conf.uccscart.characterization['entity'] = "Cart"
                else:
                    cur_conf.uccscart.characterization['entity'] = "Pole"
                cur_conf.uccscart.characterization['attribute'] = "speed"
                if inccnt > deccnt:
                    cur_conf.uccscart.characterization['change'] = 'increase'
                else:
                    cur_conf.uccscart.characterization['change'] = 'decrease'
            else:
                levelcnt[1] = 0  # remove level  as its noisy and often large but when really there its 1000s ao if here
                # even if its the max something else is goign one.
                maxi = np.argmax(levelcnt)
                cur_conf.uccscart.characterization['level'] = int(maxi)
                cur_conf.uccscart.characterization['entity'] = "Block"

                if maxi == 7 or velcnt > 10 * L4block:  # l7 will have huge numbers of blockvelocity violations
                    # should do more to figure out direction of
                    cur_conf.uccscart.characterization['attribute'] = "direction"
                    if attcart > 5:
                        cur_conf.uccscart.characterization['change'] = 'toward cart'
                    elif aimblock > 5 or parallelblock > 5:
                        cur_conf.uccscart.characterization['change'] = 'toward block'
                    else:
                        cur_conf.uccscart.characterization['change'] = 'toward location'

                if maxi == 3 or (maxi == 2 and (initcnt < diffcnt)):
                    maxi = 3
                    cur_conf.uccscart.characterization['level'] = int(maxi)
                    cur_conf.uccscart.characterization['attribute'] = "direction"
                    cur_conf.uccscart.characterization['change'] = 'toward location'
                    if (attcart > 5):
                        cur_conf.uccscart.characterization['change'] = 'toward cart'
                if maxi == 4 or L4block > (cur_conf.episode - cur_conf.scoreforKL) / 2:
                    # level 4  if most episodes since we started scoring  showin level 4 we stick with 4.. L5 gets to
                    # count every tick, l4 only toward end
                    maxi = 4  # reset to 5 does not overwrite
                    cur_conf.uccscart.characterization['level'] = int(4)
                    cur_conf.uccscart.characterization['attribute'] = "size"
                    if (cur_conf.trialchar.count("LL4 BLock Size decreased") < cur_conf.trialchar.count(
                            "LL4 BLock Size increased")):
                        cur_conf.uccscart.characterization['change'] = 'increase'
                    else:
                        cur_conf.uccscart.characterization['change'] = 'decrease'
                elif maxi == 2:
                    cur_conf.uccscart.characterization['attribute'] = "speed"
                    if inccnt > deccnt:
                        cur_conf.uccscart.characterization['change'] = 'increase'
                    else:
                        cur_conf.uccscart.characterization['change'] = 'decrease'
                elif maxi == 5:
                    cur_conf.uccscart.characterization['attribute'] = "direction"
                    if attcart > 5:
                        cur_conf.uccscart.characterization['change'] = 'toward cart'
                    elif aimblock > 5 or parallelblock > 5:
                        cur_conf.uccscart.characterization['change'] = 'toward block'
                    else:
                        cur_conf.uccscart.characterization['change'] = 'toward location'
                elif maxi == 6:
                    # should do more to figure out direction impact...  maybe use direction if falling tooo fast or
                    # slow.  ALso friction could be here..
                    cur_conf.uccscart.characterization['attribute'] = "gravity"
                    if inccnt > deccnt:
                        cur_conf.uccscart.characterization['change'] = 'increase'
                    else:
                        cur_conf.uccscart.characterization['change'] = 'decrease'

            if cur_conf.uccscart.characterization['level'] == 0:
                cur_conf.uccscart.characterization['level'] = None
                cur_conf.uccscart.characterization['entity'] = None
                cur_conf.uccscart.characterization['attribute'] = None
                cur_conf.uccscart.characterization['change'] = None

        cur_conf.trialchar += cur_conf.logstr  # save char string without any added summarization so we can compute
        # over it.

        if not finalepisode:
            cur_conf.summary += "@@@@@ Interum Output String Characterization for  world change prob: " + str(
                cur_conf.worldchangedacc) + " "
        else:
            # if final, truncate the characterstring so its just the final data
            if cur_conf.worldchangedacc < .5:
                cur_conf.summary += "##### @@@@@ Ending Characterization of potential observed novelities, but did " \
                                    "not declare world novel  with  world change prob: " + str(
                    cur_conf.worldchangedacc)
                if cur_conf.worldchangedacc >= .5:
                    cur_conf.summary += "##### @@@@@  Ending Characterization of observed novelities in novel world   " \
                                        "with  world change prob: " + str(
                        cur_conf.worldchangedacc)
        if initcnt > diffcnt:
            cur_conf.summary += "Inital world off and "
        if diffcnt > initcnt:
            cur_conf.summary += " Dynamics of world off and "
        if blockcnt > polecnt and blockcnt > cartcnt:
            cur_conf.summary += " Dominated by Blocks with"
        if cartcnt > polecnt and cartcnt > blockcnt:
            cur_conf.summary += " Dominated by Cart with"
        if polecnt > cartcnt and polecnt > blockcnt:
            cur_conf.summary += " Dominated by Pole with"
        cur_conf.summary += " Velocity Violations " + str(velcnt)
        cur_conf.summary += "; Agent Velocity Violations " + str(blockvelcnt)
        cur_conf.summary += "; Cart Total Violations " + str(cartcnt)
        cur_conf.summary += "; Pole Total Violations " + str(polecnt)
        cur_conf.summary += "; Speed/position decrease Violations " + str(deccnt)
        cur_conf.summary += "; Speed/position increase Violations " + str(inccnt)
        cur_conf.summary += "; Attacking Cart Violations " + str(attcart)
        cur_conf.summary += "; Blocks aiming at blocks " + str(aimblock)
        cur_conf.summary += "; Coordinated block motion " + str(parallelblock)
        cur_conf.summary += "; Agent Total Violations " + str(blockcnt + parallelblock + attcart + blockvelcnt)
        for i in range(1, 9):
            cur_conf.summary += "; L" + str(i) + ":=" + str(levelcnt[i])
        cur_conf.summary += ";  Violations means that aspect of model had high accumulated EVT model probability of " \
                            "exceeding normal training "
        if failcnt > 10:
            cur_conf.summary += "Uncontrollable dynamics for unknown reasons, but clearly novel as failure frequencey " \
                                "too high compared to training "
        if not outputstats:
            cur_conf.summary += "#####"

    if cur_conf.previous_wc > cur_conf.worldchangedacc:
        cur_conf.debugstring = "   worldchanged went down, line 1458" + str(cur_conf.previous_wc) + " " + str(
            cur_conf.worldchangedacc)
        print(cur_conf.debugstring)
        cur_conf.worldchangedacc = cur_conf.previous_wc

    if cur_conf.worldchangedacc > cur_conf.previous_wc:
        cur_conf.previous_wc = cur_conf.worldchangedacc
    elif cur_conf.worldchangedacc < cur_conf.previous_wc:
        cur_conf.worldchangedacc = cur_conf.previous_wc

    print('Dend# {}  Logstr={} {} Prob={} {} Scores= {}   '.format(cur_conf.episode, cur_conf.logstr,
                                                                   "\n",
                                                                   [round(num, 2) for num in cur_conf.problist[:40]],
                                                                   "\n",
                                                                   [round(num, 2) for num in cur_conf.scorelist[:40]]))

    return cur_conf.worldchangedacc
