
#####!!!!!##### Start INDEPNDENT CODE for EVT-
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

#reversed wbl for maxim fitting
def rwcdf(self,x,iloc,ishape,iscale):
    if(x-iloc< 0) : prob = 0        
    else: prob = 1-math.pow(math.exp(-(x-iloc)/iscale),ishape)
    #if(prob > 1e-4): print("in rwcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob

    #abs wbl for unsided fitting    
def awcdf(self,x,iloc,ishape,iscale):
    prob = 1-math.pow(math.exp(-abs(x-iloc)/iscale),ishape)
    #if(prob > 1e-4): print("in awcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob
    
    
#regualr wbl for minimum fits
def wcdf(self,x,iloc,ishape,iscale):
    if(iloc-x< 0) : prob = 0
    else: prob = 1-math.pow(math.exp(-(iloc-x)/iscale),ishape)
    #if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob

#####!!!!!##### End Doimain Independent CODE for EVT-


def world_change_prob(self,settrain=False):

        # don't let first episodes  impact world change.. need stabilsied scores/probabilites.. skipping work here also makes it faster
        if(self.episode< self.scoreforKL):
            self.worldchangedacc = 0
            self.worldchangeblend = 0
            self.previous_wc = 0                       
            return self.worldchangedacc            

            

        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1193"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc

        self.summary ="" #reset summary to blank
            
        

        
        mlength = len(self.problist)
        mlength = min(self.scoreforKL,mlength)
        # we look at the larger of the begging or end of list.. world changes most obvious at the ends. 

        window_width=11
        #look at list of performacne to see if its deviation from training is so that is.. skip more since it needs to be stable for window smoothing+ mean/variance computaiton
        PerfKL =    0        
        if (len(self.perflist) >(self.scoreforKL+window_width) and len(self.perflist) < 3* self.scoreforKL ):  
            #get smoothed performance 
            cumsum_vec = np.cumsum(np.insert(self.perflist, 0, 0))
            smoothed = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
            pmu = np.mean(smoothed[:-self.scoreforKL])  # we skip first/iniiprob... it is used elsehwere. 
            psigma = np.std(smoothed[:-self.scoreforKL])
            
#            if(pmu <  self.mean_perf or pmu >  self.mean_perf +  self.stdev_perf):     #if we want only  KL for those what have worse performance or much better                
            if(pmu >  self.mean_perf ):     #if we want only  KL for those what have worse performance or much better
                PerfKL = self.kullback_leibler(pmu, psigma, self.mean_perf, self.stdev_perf)
                self.debugstring = '   PerfKL {} {} {} {} ={} ,'.format(pmu, psigma, self.mean_perf, self.stdev_perf, round(PerfKL,3))
                print(self.debugstring)
                
            # If there is still too much variation (too many FP) in the variance in the small window so we use stdev and just new mean this allows smaller (faster) window for detection. 
            # PerfKL = self.kullback_leibler(pmu, self.stdev_perf, self.mean_perf, self.stdev_perf)
        else:
            PerfKL = 0

        
        if( self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1222"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc

        

        if(mlength > 1) :
            mu = np.mean(self.problist[0:mlength-1])
            sigma = np.std(self.problist[0:mlength-1])
        else:
            mu = sigma = 0
            self.debugstring = '   ***Zero Lenth World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} val {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,5),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
            print(self.debugstring)
            self.logstr +=  "&" + self.debugstring
            
            self.worldchanged = self.worldchangedacc
            return max(self.worldchangedacc,self.previous_wc);
        
       
        if(settrain):
           self.mean_train = mu;
           self.stdev_train = sigma;
           print("Set  world change train mu and sigma", mu, sigma)
           self.logstr +=  "&" + "Set  world change train mu and sigma" + str(mu) + str(sigma) +" saying world_change = 0"
           self.worldchanged = 0
           return max(self.worldchangedacc,self.previous_wc);
        
        if( self.mean_train == 0):
            self.mean_train = 0.004
            self.stdev_train = 0.009
            self.dynam_prob_scale = 1  # probably do need to scale but not tested sufficiently to see what it needs.

        if(mu > self.mean_train):
            self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)
        else:
            self.KL_val=0            
        self.debugstring = '   ***Short World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
            round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
            round(self.stdev_train,3) ,round(self.KL_val,5), round(self.KL_threshold,5), "\n", [round(num,2) for num in self.scorelist])
        if (self.debug):
            print(self.debugstring)
           
        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1260"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc


        dprob = perfprob = 0                # don't allow on short runs.. dynamics and performance are off            

        if (len(self.problist) < 198):   #for real work but short list
            self.consecutivesuccess=0            
            self.failcnt += 1
            if( self.consecutivefail >0):
                self.consecutivefail = min(self.consecutivefail+1, self.maxconsecutivefailthresh+2)
                if(self.consecutivefail > self.maxconsecutivefail):
                    self.maxconsecutivefail = self.consecutivefail
                    if(self.maxconsecutivefail > self.maxconsecutivefailthresh):
                        self.worldchangedacc = 1
                        self.logstr +=  "&" + "Step " + str(self.tick) + "#####? Uncontrollable world -- too many consecutive failures.  Guessing actions were remapped/perturbed but will take a while to confirm ##### "                         

            else: self.consecutivefail=1
            if(mu > self.mean_train):
                self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)
            else:
                self.KL_val=0                
                
            self.debugstring = '   ***Short World Change Acc={}, Failcnt= {} Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}        scores{}'.format(
                round(self.worldchangedacc,3),self.failcnt, [round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
                round(self.stdev_train,3) ,round(self.KL_val,5), round(self.KL_threshold,5), "\n", [round(num,2) for num in self.scorelist])
            if (self.debug):
                print(self.debugstring)


        else:
            self.consecutivefail=0
            self.consecutivesuccess += 1
            if(self.consecutivesuccess > self.maxconsecutivesuccess):
                self.maxconsecutivesuccess = self.consecutivesuccess

            
                
        
            
            if (sigma == 0):
                if (mu == self.mean_train):
                    self.debugstring = '      BadSigma World Change Acc={}, Prob ={},mu={}, sigmas {}, mean {} stdev{} KLval {} thresh {} {}         scores{}'.format(
                        round(self.worldchangedacc,3),[round(num,2) for num in self.problist],round(mu,3), round(sigma,3), round(self.mean_train,3),
                        round(self.stdev_train,3) ,round(self.KL_val,3), round(self.KL_threshold,3), "\n", [round(num,2) for num in self.scorelist])
                    print(self.debugstring)
                    self.logstr +=  "&" + self.debugstring
                    return max(self.worldchangedacc,self.previous_wc);

            else:
                sigma = self.stdev_train


            if(mu < self.mean_train):   #no point computing if world differences are smaller, it may be "much" smaller but that is okay
                self.KL_val = 0   
            else: 
                self.KL_val = self.kullback_leibler(mu, sigma, self.mean_train, self.stdev_train)


            #KLscale = (self.num_epochs + 1 - self.episode / 2) / self.num_epochs  # decrease scale (increase sensitvity)  from start 1 down to  1/2
            #        KLscale = min(1, 4*(1 + self.episode) / num_epochs)  # decrease scale (increase sensitvity)  from start 1 down to  1/2
            KLscale = 1
            dprob = min(1.0, ((KLscale * self.KL_val) )) 
            perfprob = min(1.0, self.PerfScale * PerfKL)  #make this smaller since it is slowly varying and  added every time.. less sensitive (good for FP avoid, sloer l

        #if we had  collisons and not consecuretive faliures, we don't use this episode for dynamic probability .. collisions are not well predicted
        #tlen = min(self.scoreforKL,len(self.uccscart.char))
                   
        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1325"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc
            

        #random collisions can occur and they destroy probability computation so ignore them
        if( self.logstr.count("CP") > 4)  and ( "attack" not in self.logstr) and (self.consecutivefail < 4):
            prob = 0
            dprob=.0009990
            perfprob=0
            print("Debug found ", self.logstr.count("CP"), "CPs in string")
        else:
            prob = min(1,max(dprob,perfprob)) # use max of dynamic and long-term performance probabilities.
            print("Debug did not find many CP ", self.logstr.count("CP"), " in string. Prob=",str(prob))            


        #     # infrequent checkto outputting cnts for setinng up wbls for actual cnts to use to see if we whould update world change
        if(((self.episode-10)  ==  self.scoreforKL) or ((self.episode-10)  ==  2*self.scoreforKL)):
            cntval= np.zeros(15)
            cntprob= np.zeros(21)            
            i=0
            scale= 1./int((self.episode) )
            cntval[i] = initcnt = scale*self.trialchar.count("init")
            i+= 1; cntval[i] = blockcnt = scale*self.trialchar.count("Block")
            i+= 1; cntval[i] =blockvelcnt = scale*self.trialchar.count("Block Vel")                
            i+= 1; cntval[i] =polecnt = scale*self.trialchar.count("Pole")
            i+= 1; cntval[i] =cartcnt = scale*self.trialchar.count("Cart")
            i+= 1; cntval[i] =smallcnt = scale*self.trialchar.count("Cart")
            i+= 1; cntval[i] =largecnt = scale*self.trialchar.count("Cart")                                
            i+= 1; cntval[i] =diffcnt = scale*self.trialchar.count("diff")
            i+= 1; cntval[i] =velcnt = scale*self.trialchar.count("Vel")                                
            i+= 1; cntval[i] =failcnt = scale*self.trialchar.count("High")
            i+= 1; cntval[i] =attcart = scale*self.trialchar.count("attacking cart")
            i+= 1; cntval[i] =aimblock = scale*self.trialchar.count("aiming")
            i+= 1; cntval[i] =parallelblock= scale*self.trialchar.count("parallel")
            levelcnt=np.zeros(9)
            i=0
            i+= 1; levelcnt[i] =L1= scale*self.trialchar.count("LL1")
            i+= 1; levelcnt[i] =L2block= scale*self.trialchar.count("LL2")
            i+= 1; levelcnt[i] =L3block= scale*self.trialchar.count("LL3")
            i+= 1; levelcnt[i] =L4block= scale*self.trialchar.count("LL4")
            i+= 1; levelcnt[i] =L5block= scale*self.trialchar.count("LL5")
            i+= 1; levelcnt[i] =L6block= scale*self.trialchar.count("LL6")
            i+= 1; levelcnt[i] =L7block= scale*self.trialchar.count("LL7")
            i+= 1; levelcnt[i] =L8block= scale*self.trialchar.count("LL8")


            '''  New String matching for json output 
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
            
            '''

            cntwbl = DATA.cntwbl

            cntmax=0
            for i in range(12):
                cntprob[i] = self.wcdf(-cntval[i],cntwbl[i,1],cntwbl[i,0],cntwbl[i,2])
                cntmax = max (cntmax,cntprob[i])


            if(cntmax > .1): cntmax=.1;  #limit impact this this is really a cumulative test on things we have already seen
            
            #approximate wibul for  count of SA
            cntprob[13] = self.wcdf(152.73 - cntval[13],0.1843,2.01214,36.5311)
            cntmax += cntprob[13]
            
            if(prob < cntmax):
                self.logstr += 'Using detect as prob. detectcnts: {} dectprob {} '.format(cntval,cntprob)
                prob = max(prob,cntmax)
            else: self.logstr += 'detectcnts: {} dectprob {} '.format(cntval,cntprob)
                

                        
        if (len(self.problist) < self.scoreforKL):
            self.worldchanged = prob * len(self.problist)/(self.scoreforKL)
        elif (len(self.problist) < 2* self.scoreforKL):
            self.worldchanged = prob            
        else: # if very long list, KL beceomes too long, its more likely to be higher from random agent crashing into pole so we the impact
            self.worldchanged = prob * (2*self.scoreforKL)/len(self.problist)
            

            
        #only do blockmin/max if we did a long enough  note this does not need any blending or scoreforKL since its an absolute novelty to have this happen
        # if the blockmax/min (in general from past episode) from past are not normal add them to the worldchange acce.  Again should be EVT based but not enough training yet.
        if (self.episode > (self.scoreforKL) and self.tick >170 and self.blockvelmax > 5):   # if we have been through enough episodes and enough steps and blocks moved enough
            minmaxupdate = abs(self.blockmin-self.normblockmin) + abs(self.blockmax-self.normblockmax)
            if(minmaxupdate > .2):
                if(self.blockmin < self.normblockmin or self.blockmax> self.normblockmax):
                    self.logstr += '&& P2+ LL4 BLock Size decreased. minmaxupdate='+ str(minmaxupdate)
                elif(self.blockmin > self.normblockmin or self.blockmax< self.normblockmax):
                    self.logstr += '&& P2+ LL4 BLock Size increased ' + str(minmaxupdate)
                print('  minmaxupdate {} min {} max {} normal {} {}  ,'.format(minmaxupdate,self.blockmin,self.blockmax,self.normblockmin,self.normblockmax)        )
                prob = prob+ .3
                self.worldchanged += .3              
                self.debugstring = '   Prob {} after Level LL4 Size Change minmaxupdate {} min {} max {} normal {} {} VelMax {}  ,'.format(prob,minmaxupdate,self.blockmin,self.blockmax,self.normblockmin,self.normblockmax,self.blockvelmax)
                print(self.debugstring)
                self.logstr +=  "&" + self.debugstring                


        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1314"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)
            self.logstr +=  "&" + self.debugstring                            
            self.worldchangedacc=self.previous_wc                              


        outputstats=False
            

#####!!!!!##### end GLue CODE for EVT

#####!!!!!#####  Domain Independent code tor consecurtiv efailures
        failinc = 0
        #if we are beyond KL window all we do is watch for failures to decide if we world is changed

            
        if(self.episode > self.scoreforKL+1 and self.episode < 3* self.scoreforKL):
            faildiff = self.failcnt/(self.episode+1)-self.failfrac            
            if(faildiff > 0):
                self.logstr +=  "&" + "Step " + str(self.tick) + "High FailFrac=" + str(self.failcnt/(self.episode+1))
                failinc = max(0,  ((faildiff)*self.failscale)) 
                failinc *= min(1,(self.episode - self.scoreforKL)/self.scoreforKL)   #Ramp it up slowly as its more unstable when it first starts at scoreforKL
                failinc = min(1,failinc)

            #world change blend  can go up or down depending on how probablites vary.. goes does allows us to ignore spikes from uncommon events. as the bump i tup but eventually go down. 
            if(prob < .5 and  self.worldchangedacc <.5) : # blend wo  i.e. decrease world change accumulator to limit impact of randome events
                #self.worldchangeblend = min(self.worldchangedacc * self.blenddownrate, (self.blenddownrate *self.worldchanged + (1-self.blenddownrate) * self.worldchangeblend ))
                self.worldchangedacc = min(1,self.worldchangedacc*self.worldchangeblend)            
                self.debugstring = "BlendDown "                

            else:
                #worldchange acc once its above .5 it cannot not go down.. it includes max of old value..
                self.worldchangeblend = min(1, (        self.blenduprate *self.worldchanged + (1-self.blenduprate) * self.worldchangeblend ))

                self.debugstring = "Blendup using rate " + str(self.blenduprate) + "wc/wcb="+ str(self.worldchanged) + " " + str(self.worldchangeblend)                                
                # we add in an impusle each step if the first step had initial world change.. so that accumulates over time

                if(len(self.problist) > 0 ) :
                    self.worldchangedacc = min(1,self.problist[0]*self.initprobscale + (self.worldchangedacc+self.worldchanged)*min(1,self.worldchangeblend+failinc))
                else:
                    self.worldchangedacc = min(1,max(self.worldchangedacc+self.worldchanged,self.worldchangedacc*self.worldchangeblend+failinc))
                self.debugstring += '    mu={}, sig {}, mean {} stdev{}  WCacc WBlend {} {} vals {} {} thresh {} '.format(
                    round(mu,3), round(sigma,3), round(self.mean_train,3), round(self.stdev_train,3), round(self.worldchangedacc,3), round(self.worldchangeblend,3) ,round(self.KL_val,3),round(PerfKL,3),  "\n")
                print(self.debugstring)


        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged when down, line 1358"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)            
            self.worldchangedacc=self.previous_wc          

        # Until we have enough data, reset and worldchange, don't start accumulating
        if(self.episode < self.scoreforKL):
            self.worldchangedacc = 0
            self.worldchangeblend = 0            

            # normal world gets randome incremenets not bigger ones all in a row..  novel world can get many consecutitive one so we increase prob of that big an increment and expodentially based on sequence length and .
            # Parms weak as not much data in training as its too infrequently used.  Not used later in stage as random errors seem to grow with resets so consecurive random becomes more likely
        if((self.worldchangedacc - self.previous_wc) > self.minprob_consecutive and self.episode < 2* self.scoreforKL):
            if(self.consecutivewc >0):
                wcconsecutivegrowth = (self.consecutivewc/20)*  (1-self.wcdf((self.worldchangedacc - self.previous_wc),.001,.502,.13))
                print("wcconsecutivegrowth = ", wcconsecutivegrowth, self.worldchangedacc, self.previous_wc)
                self.logstr += 'World Change Consecutive {}, prev {} base {} increent {} '.format(self.consecutivewc,round(self.previous_wc,3), round(self.worldchangedacc,3),round(wcconsecutivegrowth,3))
                self.worldchangedacc = min(1, self.worldchangedacc  + wcconsecutivegrowth)
                print("EPi2 previs new world change", self.episode, self.previous_wc, self.worldchangedacc)
                
            self.consecutivewc += 1
        else:  self.consecutivewc = 0

        
#####!!!!!#####  End Domain Independent code tor consecurtiv efailures


#####!!!!!#####  Start API code tor reporting
        self.logstr += 'World Change Acc={} {} {} {}, CW={},CD={} D/KL Probs={},{}'.format(round(self.worldchangedacc,3), round(self.worldchangeblend,3),round(self.previous_wc,3),round(failinc,3), self.consecutivewc,self.dynamiccount,round(dprob,3), round(perfprob,3))


        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1389"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)            


        print("EPi previs new world change", self.episode, self.previous_wc, self.worldchangedacc)
        if(self.previous_wc < .5 and self.worldchangedacc        >= .5):
            if(self.noveltyindicator == True):
                self.logstr +=  self.hint                
                self.logstr += "#!#!#!  World change TP Detection "+ str(self.episode) + "  @@ FV= "
                self.summary += "#!#!#!  World change TP Detection "+ str(self.episode) + "  @@  "                
            elif(self.noveltyindicator == False):
                self.logstr +=  self.hint
                self.logstr += "#!#!#!  World change FP Detection "+ str(self.episode) + "  @@ FV= "
                self.summary += "#!#!#!  World change FP Detection "+ str(self.episode) + "  @@  "                                
            else: self.logstr += "#!#!#!  World change blind Detection "+ str(self.episode) + "  @@ FV= "    
            self.logstr += str(self.current_state)
            print ("   Detecting at Episode==", self.episode, " hint=",self.hint); 
            outputstats=True            

        # if world changed an dour performance is below .65  we start using avoidance reaction
        #        if(self.worldchangedacc        >= .6 and (100*self.perf/self.totalcnt) < 65) :
        #                self.uccscart.use_avoid_reaction=True            

        if(        self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1403"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)            
            self.worldchangedacc=self.previous_wc          

        outputstats=True
        finalepisode =  False
        if((self.episode+1)%99 ==0 ): finalepisode =  True
        if(finalepisode or outputstats):
            initcnt = self.trialchar.count("init")
            blockcnt = self.trialchar.count("Block")
            blockvelcnt = self.trialchar.count("Block Vel")
            blockmotion = self.trialchar.count("Block Motion")                            
            polecnt = self.trialchar.count("Pole")
            cartcnt = self.trialchar.count("Cart")
            smallcnt = self.trialchar.count("small")
            inccnt = self.trialchar.count("increase")
            deccnt = self.trialchar.count("decrease")            
            largecnt = self.trialchar.count("large")                                
            diffcnt = self.trialchar.count("diff")
            velcnt = self.trialchar.count("Vel")                                
            failcnt = self.trialchar.count("High")
            attcart = self.trialchar.count("Relational Attacking cart")
            aimblock = self.trialchar.count("Relational aiming")
            parallelblock = self.trialchar.count("Relational parallel")
            levelcnt=np.zeros(10)
            i=0
            i+= 1; levelcnt[i] =L1= self.trialchar.count("LL1")
            i+= 1; levelcnt[i] =L2block= self.trialchar.count("LL2")
            i+= 1; levelcnt[i] =L3block= self.trialchar.count("LL3")
            i+= 1; levelcnt[i] =L4block= self.trialchar.count("LL4")
            i+= 1; levelcnt[i] =L5block= self.trialchar.count("LL5")
            i+= 1; levelcnt[i] =L6block= self.trialchar.count("LL6")
            i+= 1; levelcnt[i] =L7block= self.trialchar.count("LL7")
            i+= 1; levelcnt[i] =L8block= self.trialchar.count("LL8")


            # if we get enough L5 we increase our world change estimate
            if(L5block > 50 and self.episode < 2.5*self.scoreforKL):
                self.worldchangedacc = min(1,.25+self.worldchangedacc);
                
            maxi = np.argmax(levelcnt)

            # if level is 8 is already filled in,  no need to do any filling
            if(self.uccscart.characterization['level'] == int(8) or maxi ==8):
                #8 had special code to do increase vs increasing
                self.uccscart.characterization['level']=int(8);
                self.uccscart.characterization['entity']="Block"; 
                self.uccscart.characterization['attribute']="quantity";
                if(self.logstr.count("LL8: Blocks quantity dec") > self.logstr.count("LL8: Blocks quantity inc")  ):  #if we had more than one chance its increasing
                    if(self.logstr.count("LL8: Blocks quantity dec") >2  ):  #if we had more than one chance its increasing                    
                        self.uccscart.characterization['change']='decreasing';
                    else:
                        self.uccscart.characterization['change']='decrease';                        
                else:
                    if(self.logstr.count("LL8: Blocks quantity inc") >2  ):  #if we had more than one chance its increasing                                        
                        self.uccscart.characterization['change']='increasing';
                    else:
                        self.uccscart.characterization['change']='increase';                        
            else:
                self.uccscart.characterization['level']=int(maxi)
                self.uccscart.characterization['level']=None
                self.uccscart.characterization['entity']=None
                self.uccscart.characterization['attribute']=None
                self.uccscart.characterization['change']=None           
                
                if(maxi == 1 and levelcnt[1] > 1000):
                    self.uccscart.characterization['level']=int(1);
                    if(cartcnt > polecnt):
                        self.uccscart.characterization['entity']="Cart";
                    else:
                        self.uccscart.characterization['entity']="Pole";
                    self.uccscart.characterization['attribute']="speed";
                    if(inccnt > deccnt):
                        self.uccscart.characterization['change']='increase';
                    else:
                        self.uccscart.characterization['change']='decrease';                    
                else:

                    levelcnt[1] = 0 # remove level  as its noisy and often large but when really there its 1000s ao if here even if its the max something else is goign one. 
                    maxi = np.argmax(levelcnt)
                    self.uccscart.characterization['level']=int(maxi);                            
                    self.uccscart.characterization['entity']="Block";
                            
                    if (maxi == 7  or velcnt > 10*L4block):    #l7 will have huge numbers of blockvelocity violations
                        # should do more to figure out direction of 
                        self.uccscart.characterization['attribute']="direction";
                        if(attcart > 5):
                           self.uccscart.characterization['change']='toward cart';                           
                        elif(aimblock > 5 or  parallelblock > 5 ):
                            self.uccscart.characterization['change']='toward block';
                        else:
                            self.uccscart.characterization['change']='toward location';
                    
                    if(maxi == 3 or  (maxi == 2 and (initcnt < diffcnt) )):
                        maxi =3
                        self.uccscart.characterization['level']=int(maxi);                                                                                
                        self.uccscart.characterization['attribute']="direction";
                        self.uccscart.characterization['change']='toward location';
                        if(attcart > 5):
                           self.uccscart.characterization['change']='toward cart';                                                   
                    if (maxi == 4 or  L4block> (self.episode-self.scoreforKL)/2):
                        #level 4  if most episodes since we started scoring  showin level 4 we stick with 4.. L5 gets to count every tick, l4 only toward end
                        maxi=4 #reset to 5 does not overwrite
                        self.uccscart.characterization['level']=int(4);                                                        
                        self.uccscart.characterization['attribute']="size";
                        if(self.trialchar.count("LL4 BLock Size decreased") <self.trialchar.count("LL4 BLock Size increased")):
                            self.uccscart.characterization['change']='increase';
                        else: 
                            self.uccscart.characterization['change']='decrease';
                    elif (maxi == 2):
                        self.uccscart.characterization['attribute']="speed";
                        if(inccnt > deccnt):
                            self.uccscart.characterization['change']='increase';
                        else:
                            self.uccscart.characterization['change']='decrease';                    
                    elif (maxi == 5):
                        self.uccscart.characterization['attribute']="direction";
                        if(attcart > 5):
                           self.uccscart.characterization['change']='toward cart';                           
                        elif(aimblock > 5 or  parallelblock > 5 ):
                            self.uccscart.characterization['change']='toward block';
                        else:
                            self.uccscart.characterization['change']='toward location';
                    elif(maxi == 6):
                        # should do more to figure out direction impact...  maybe use direction if falling tooo fast or
                        # slow.  ALso friction could be here..                          
                        self.uccscart.characterization['attribute']="gravity";
                        if(inccnt > deccnt):
                            self.uccscart.characterization['change']='increase';
                        else:
                            self.uccscart.characterization['change']='decrease';


                if(self.uccscart.characterization['level']==0):
                    self.uccscart.characterization['level']=None                    
                    self.uccscart.characterization['entity']=None
                    self.uccscart.characterization['attribute']=None
                    self.uccscart.characterization['change']=None           


                
            self.trialchar += self.logstr  #save char string without any added summarization so we can compute over it. 

            if(not finalepisode):
                self.summary += "@@@@@ Interum Output String Characterization for  world change prob: " +str(self.worldchangedacc) +" "            
            else:
                #if final, truncate the characterstring so its just the final data
                if(self.worldchangedacc        <.5):
                    self.summary += "##### @@@@@ Ending Characterization of potential observed novelities, but did not declare world novel  with  world change prob: " +str(self.worldchangedacc)
                    if(self.worldchangedacc        >= .5):                
                        self.summary += "##### @@@@@  Ending Characterization of observed novelities in novel world   with  world change prob: " +str(self.worldchangedacc)
            if(initcnt > diffcnt ):
                self.summary += "Inital world off and "
            if(diffcnt > initcnt ):
                self.summary += " Dynamics of world off and "                    
            if(blockcnt > polecnt and blockcnt > cartcnt ):                    
                self.summary += " Dominated by Blocks with"
            if(cartcnt > polecnt and cartcnt >  blockcnt   ):                    
                self.summary += " Dominated by Cart with"
            if(polecnt > cartcnt and polecnt > blockcnt ):                    
                self.summary += " Dominated by Pole with"
            self.summary += " Velocity Violations " + str(velcnt)                                                                                                
            self.summary += "; Agent Velocity Violations " + str(blockvelcnt)                
            self.summary += "; Cart Total Violations " + str(cartcnt)
            self.summary += "; Pole Total Violations " + str(polecnt)
            self.summary += "; Speed/position decrease Violations " + str(deccnt)
            self.summary += "; Speed/position increase Violations " + str(inccnt)                                                                                                            
            self.summary += "; Attacking Cart Violations " + str(attcart)
            self.summary += "; Blocks aiming at blocks " + str(aimblock)                                                                                                            
            self.summary += "; Coordinated block motion " + str(parallelblock)
            self.summary += "; Agent Total Violations " + str(blockcnt + parallelblock + attcart + blockvelcnt)
            for i in range(1,9):
                self.summary += "; L" + str(i) + ":=" + str(levelcnt[i])
            self.summary += ";  Violations means that aspect of model had high accumulated EVT model probability of exceeding normal training  "
            if(failcnt > 10):
                self.summary += " Uncontrollable dynamics for unknown reasons, but clearly novel as failure frequencey too high compared to training"
            if(not outputstats):                
                self.summary += "#####"

                



        if(self.previous_wc > self.worldchangedacc):
            self.debugstring = "   worldchanged went down, line 1458"+ str(self.previous_wc) + " " + str(self.worldchangedacc)
            print(self.debugstring)
            self.worldchangedacc=self.previous_wc          
                
        if(self.worldchangedacc > self.previous_wc):  self.previous_wc = self.worldchangedacc
        elif(self.worldchangedacc < self.previous_wc):  self.worldchangedacc = self.previous_wc 

        print('Dend# {}  Logstr={} {} Prob={} {} Scores= {}   '.format( self.episode,  self.logstr,
                                                                        "\n", [round(num, 2) for num in self.problist[:40]], 
                                                                        "\n", [round(num, 2) for num in self.scorelist[:40]]))            
        

        return self.worldchangedacc;