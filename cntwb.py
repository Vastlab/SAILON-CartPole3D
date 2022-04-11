from scipy.stats import weibull_min
import numpy as np
import math
import pdb

def rwcdf(x,iloc,ishape,iscale):
    if(iloc-x< 0) : prob = 0
    else: prob = 1-math.pow(math.exp(-(iloc-x)/iscale),ishape)
    #        if(prob > 1e-4): print("in wcdf",round(prob,6),x,iloc,ishape,iscale)
    return prob

b

cnts39=np.array([[ 3, 3, 0, 0, 2, 2, 2, 2, 2, 0, 3, 0],
                 [ 2, 2, 0, 0, 3, 3, 3, 3, 3, 0, 0, 2],
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 4, 4, 0, 0, 3, 3, 3, 3, 3, 0, 0, 4],
                 [ 5, 5, 5, 0, 2, 2, 2, 2, 7, 0, 0, 0],
                 [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                 [ 3, 3, 3, 0, 2, 2, 2, 2, 5, 0, 0, 0],
                 [ 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0],
                 [ 7, 7, 3, 0, 1, 1, 1, 1, 4, 0, 0, 4],
                 [ 3, 3, 0, 0, 1, 1, 1, 1, 1, 0, 3, 0],
                 [ 4, 4, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0],
                 [ 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [ 5, 5, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 1, 1, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1],
                 [ 0, 3, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                 [ 2, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                 [ 3, 3, 0, 1, 2, 2, 2, 3, 3, 1, 3, 0],
                 [ 5, 5, 5, 0, 3, 3, 3, 3, 8, 0, 0, 0],
                 [ 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                 [ 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 0],
                 [ 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                 [ 3, 3, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                 [ 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0],
                 [ 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                 [ 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],                 
                 [ 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0]])


cnts100 = np.array([    [14, 10, 1, 5, 5, 5, 6, 16, 0, 4, 0, 0],
                       [5, 5, 3, 2, 3, 3, 3, 5, 8, 0, 0, 2],
                       [0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0],
                       [2, 2, 0, 0, 5, 5, 5, 5, 5, 0, 2, 0],
                       [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                       [7, 7, 3, 0, 3, 3, 3, 3, 6, 0, 0, 4],
                       [19, 13, 0, 2, 2, 2, 2, 15, 0, 3, 3, 0],
                       [0, 0, 0, 2, 3, 3, 3, 5, 5, 0, 0, 0],
                       [3, 3, 3, 4, 2, 2, 2, 6, 9, 0, 0, 0],
                       [6, 6, 2, 3, 2, 2, 2, 5, 7, 0, 0, 4],
                       [1, 11, 3, 0, 4, 4, 4, 4, 7, 0, 0, 8],
                       [7, 7, 0, 2, 4, 4, 4, 6, 6, 0, 6, 1],
                       [5, 5, 4, 0, 0, 0, 0, 0, 4, 0, 0, 1],
                       [8, 8, 0, 5, 3, 3, 3, 8, 8, 0, 0, 8],
                       [3, 13, 5, 1, 2, 2, 2, 3, 8, 0, 5, 3],
                       [5, 5, 5, 1, 0, 0, 0, 1, 6, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [7, 0, 4, 6, 6, 6, 10, 10, 0, 6, 1, 0],
                       [0, 3, 0, 3, 2, 2, 2, 5, 5, 0, 0, 0],
                       [0, 11, 1, 5, 2, 2, 2, 7, 8, 0, 1, 8],
                       [3, 3, 0, 1, 2, 2, 2, 3, 3, 1, 3, 0],
                       [8, 8, 8, 0, 3, 3, 3, 3, 11, 0, 0, 0],
                       [3, 3, 0, 0, 2, 2, 2, 2, 2, 0, 2, 1],
                       [9, 9, 7, 1, 2, 2, 2, 3, 10, 0, 2, 0],
                       [15, 10, 0, 1, 1, 1, 1, 11, 0, 1, 4, 0],
                       [3, 3, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                       [0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
                       [6, 6, 2, 2, 3, 3, 3, 5, 6, 0, 0, 4],
                       [1, 1, 0, 2, 6, 6, 6, 8, 8, 0, 1, 0]
                  ])

cntcols=12
cntwbl = np.zeros((cntcols,3))
cnts100= (cnts100)/100.
cnts39 = cnts39 / 49.
maxes = np.max(cnts100,axis=0)+.00000001 #so we can flip to use min
rmaxes = np.max(cnts39,axis=0) #so we can flip to use min
for col in range(cntcols):
    cntwbl[col,0],cntwbl[col,1],cntwbl[col,2] = weibull_min.fit(-cnts100[:,col],floc=-maxes[col])


cntprob = np.zeros(12)

pdb.set_trace()
for i in range(12):
    cntprob[i] = rwcdf(-rmaxes[i],cntwbl[i,1],cntwbl[i,0],cntwbl[i,2])

print("probs", cntprob)
print("max-rmax", maxes- rmaxes)    
    

print("saving cnt fit data")
print(cntwbl)
np.save("cntfit.npy",  cntwbl)
      

    
    
