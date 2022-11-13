import importlib.util
import random
from datetime import datetime


class CurrentConfig:
    __INSTANCE = None

    # calibrated values for KL for cartpole wth one-step lookahead
    KL_threshold = 1
    KL_val = 0
    # we only use the first sets of scores for KL because novels worlds close when balanced
    scoreforKL = 20
    num_epochs = 200
    num_dims = 4
    num_blocks = None
    scalelargescores = 20
    # takes a while for some random starts to stabilise so don't reset too early as it
    # reduces world change sensitivity.  Effective min is 1 as need at least a prior state to get prediction.
    skipfirstNscores = 1
    # both max for per episode individual prob as well as prob scale.
    maxinitprob = 4
    current_state = None
    #        statelist=np.empty(200, dtype=object)

    blockmin = 999
    blockmax = -999
    blockvelmax = -999
    normblockmin = 999
    normblockmax = -999

    # if we see this many in a row we declare world changed as we never see even 3 in training
    maxconsecutivefailthresh = 4

    # we penalize for high failure rantes..  as  difference (faildiff*failscale) )
    failscale = 8.0  # How we scale failure fraction.. can be larger than one since its fractional differences and
    # genaerally < .1 mostly < .05
    # Max fail fraction,  when above  this we start giving world-change probability for  failures
    failfrac = .25

    # because of noisy simulatn and  many many fields and its done each time step, we limit how much this can add per
    # time step
    maxdynamicprob = .175
    # because of broken simulator we get randome bad value in car/velocity. when we detect them we limit their impact
    # to this ..
    maxclampedprob = .005
    clampedprob = maxclampedprob
    # we scale prob from cart/pole because the environmental noise, if we fix it this will make it easire to adapt .
    cartprobscale = .25
    # we scale prob from initial state by this amount (scaled as consecuriteinit increases) and add world accumulator
    # each time. No impacted by blend this balances risk from going of on non-novel worlds
    initprobscale = 1.0
    consecutiveinit = 0  # if get consecutitve init failures we keep increasing scale
    dynamiccount = 0  # if get consecutitve dynamic failures we keep increasing scale
    # if get consecutitve world change overall we keep increasing scale
    consecutivewc = 0

    # Large "control scores" often mean things are off, since we never know the exact model we reset when scores get
    # too large in hopes of  better ccotrol
    scoretoreset = 1000

    # smoothed performance plot for detection.. see perf score.py for compuation.  Major changes in control mean these
    # need updated
    perflist = []
    mean_perf = 0.8883502538071065
    stdev_perf = 0.0824239133691708
    # How much do we weight Performance KL prob.  make this small since it is slowly varying and added every episode.
    # Small is  less sensitive (good for FP avoid, but yields slower detection).
    PerfScale = 0.15

    consecutivesuccess = 0
    consecutivefail = 0
    maxconsecutivefail = 0
    maxconsecutivesuccess = 0
    minprob_consecutive = .1
    mindynprob = .01
    dynblocksprob = 0
    # should be not be larger than maxdynamicprob
    assert (minprob_consecutive <= maxdynamicprob)
    tick = 0

    maxcarlen = 25600

    # TODO: change evm data dimensions
    if num_dims == 4:
        mean_train = 0
        stdev_train = 0.0
        # need to scale since EVM probability on perfect training data is .5  because of small range and tailsize
        dynam_prob_scale = 2
    else:
        # mean_train = .198   #these are old values from Phase 1 2D cartpole..  for Pahse 2 3D we compute frm a
        # training run. stdev_train = 0.051058052318592555
        mean_train = 0.004
        stdev_train = 0.009
    # probably do need to scale but not tested sufficiently to see what it needs.
    dynam_prob_scale = 1

    cnt = 0
    framecnt = 0
    saveframes = False
    saveprefix = random.randint(1, 10000)

    worldchanged = 0
    worldchangedacc = 0
    previous_wc = 0
    # fraction of new prob we use when blending up..  It adapts over time
    blenduprate = 1
    # fraction of new prob we use when blending down..  should be less than beld up rate.  No use of max
    blenddownrate = .1

    failcnt = 0
    worldchangeblend = 0
    # from WSU "train".. might need ot make this computed.
    # mean_train=  0.10057711735799268
    # stdev_train = 0.00016
    problist = []
    scorelist = []
    maxprob = 0
    meanprob = 0
    noveltyindicator = None
    correctcnt = 0
    rcorrectcnt = 0
    totalcnt = 0
    perf = 0
    perm_search = 0
    prev_action = 0
    prev_state = None
    prev_predict = None

    # expected_backtwo = np.zeros(4)
    episode = 0
    trial = 0
    given = False

    # statelist=np.empty(200, dtype=object)
    debug = False
    debug = True
    debugstring = ""
    logstr = ""
    summary = ""
    hint = ""
    trialchar = ""

    imax = imin = imean = istd = ishape = iscale = None
    dmax = dshape = dscale = dmean = dstd = None

    """
        if (True):
        # uccscart.reset(actual_state)
        ldist = dist_cal.line_to_line_dist(np.array([0, 0, 1]), np.array(
            [0, 0, -1]), np.array([0, 1, 0]), np.array([0, -1, 0]))
        ldist2 = dist_cal.line_to_line_dist(np.array([0, 0, 1]), np.array(
            [0, 0, -1]), np.array([0, 1, 0]), np.array([0, -1, .001]))
        pdist = dist_cal.point_to_line_dist(
            np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 1, 0]))
        pdist2 = dist_cal.point_to_line_dist(
            np.array([0, 0, 0]), np.array([0, 2, 0]), np.array([0, 1, 0]))
        print("line distances", ldist, ldist2, "Pointdist", pdist, pdist2)
    """

    # Create prediction environment
    env_location = importlib.util.spec_from_file_location('CartPoleBulletEnv', 'cartpolepp/UCart.py')
    env_class = importlib.util.module_from_spec(env_location)
    env_location.loader.exec_module(env_class)

    myconfig = dict()
    myconfig['start_zeroed_out'] = False

    # Package params here
    params = dict()
    params['seed'] = 0
    params['config'] = myconfig
    # params['path'] = "WSU-Portable-Generator/source/partial_env_generator/envs/cartpolepp"
    params['path'] = "./cartpolepp"
    params['use_img'] = False
    params['use_gui'] = False

    uccscart = env_class.CartPoleBulletEnv(params)
    uccscart.path = "./cartpolepp"

    starttime = datetime.now()
    cumtime = starttime - datetime.now()

    def __new__(cls, *args, **kwargs):
        if cls.__INSTANCE is None:
            cls.__INSTANCE = super().__new__(cls)
        return cls.__INSTANCE
