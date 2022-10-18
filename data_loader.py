import numpy as np


dimansion_name = [" x Cart", " y Cart", " z Cart",  " x Cart Vel", " y Cart Vel", " z Cart Vel ",  " x Pole", " y Pole", " z Pole", " w Pole",  " x Pole Vel", " y Pole Vel", " z Pole Vel", " z Block ", " y Block", " z Block",  " x Block Vel", " y Block Vel", " z Block Vel", " 1x Wall",
                  " 1y Wall", " 1z Wall", " 2x Wall", " 2y Wall", " 2z Wall", " 3x Wall", " 3y Wall", " 3z Wall", " 4x Wall", " 4y Wall", " 4z Wall", " 5x Wall", " 5y Wall", " 5z Wall", " 6x Wall", " 6y Wall", " 6z Wall", " 8x Wall", " 8y Wall", " 8z Wall", " 9x Wall", " 9y Wall", " 9z Wall"]

# computed
imax = np.array([2.986346e+00, 2.980742e+00, 0.000000e+00, 1.984800e-02,
                 1.971800e-02, 0.000000e+00, 9.936000e-03, 9.933000e-03,
                 9.938000e-03, 9.999980e-01, 2.725000e-02, 2.736900e-02,
                 5.800000e-04, 4.090533e+00, 4.099369e+00, 9.122096e+00,
                 9.842307e+00, 9.854365e+00, 9.848242e+00, 5.000000e+00,
                 5.000000e+00, 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00, 0.000000e+00,
                 5.000000e+00, 5.000000e+00, 0.000000e+00, 5.000000e+00,
                 5.000000e+00, 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00, 1.000000e+01,
                 5.000000e+00, 5.000000e+00, 1.000000e+01])

# adjusted based on code..
imax = np.array([3.0+00, 3.0+00, 0.000000e+00,  # cart pos
                 2.e-02, 2.e-02, 0.000000e+00,  # cart vel
                 1.0e-02, 1.0e-02, 1.0e-02, 1.0e-00,  # pole pos quat
                 2.8e-02, 2.8e-02, 5.800000e-01,  # pole vel
                 4.50+00, 4.50+00, 9.50,  # block pos
                 11.5, 11.5, 11.5,  # block vel.. they can speed up before first time we see them.. program limit is 10 but gravity can accelerate
                 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01])

imin = np.array([-3.0+00, -3.0+00, 0.000000e+00,  # cart pos
                 -2.e-02, -2.e-02, 0.000000e+00,  # cart vel
                 -1.0e-02, -1.0e-02, -1.0e-02, 9.999970e-01,  # pole pos quat
                 -2.8e-02, 2.8e-02, 5.800000e-04,  # pole vel
                 -4.50533e+00, -4.500533e+00, 0.05533,  # block pos
                 4.5, 4.5, 4.5,  # block vel  based on  programmed values.. though as it can drop with gravity on first step which we we don't see so
                 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 0.000000e+00, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01, 5.000000e+00, 5.000000e+00,
                 1.000000e+01])

initwbl = np.array([[[1.38261180e+00, 0.00000000e+00, 5.97191369e-03],
                     [3.00475911e+00, 0.00000000e+00, 3.95621482e-03],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.87280624e+00, 0.00000000e+00, 1.63449012e-04],
                     [1.46913032e+00, 0.00000000e+00, 8.67032946e-05],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.81339070e+00, 0.00000000e+00, 2.99451005e-05],
                     [2.69843300e+00, 0.00000000e+00, 3.59685725e-05],
                     [1.74449197e+00, 0.00000000e+00, 1.80368642e-05],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.59112566e+00, 0.00000000e+00, 1.29455765e-03],
                     [1.51114138e+00, 0.00000000e+00, 1.24665112e-03],
                     [1.74512625e+00, 0.00000000e+00, 4.33527749e-05],
                     [1.71965636e+00, 0.00000000e+00, 1.14079390e-02],
                     [1.83879399e+00, 0.00000000e+00, 1.70432620e-02],
                     [1.11644671e+00, 0.00000000e+00, 1.99680658e-02],
                     [1.39686903e+00, 0.00000000e+00, 1.24734419e-02],
                     [1.84194195e+00, 0.00000000e+00, 8.72732910e-03],
                     [1.81573511e+00, 0.00000000e+00, 1.00200122e-02],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                    [[1.24454719e+03, 0.00000000e+00, 5.97799946e+00],
                     [8.91362642e+02, 0.00000000e+00, 5.96469200e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.04886210e+03, 0.00000000e+00, 3.97617801e-02],
                     [1.10240472e+03, 0.00000000e+00, 3.96653361e-02],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [2.76761926e+03, 0.00000000e+00, 1.98806715e-02],
                     [3.97782525e+03, 0.00000000e+00, 1.98873591e-02],
                     [5.46739493e+03, 0.00000000e+00, 1.98999663e-02],
                     [3.70416373e+01, 0.00000000e+00, 1.27680956e-04],
                     [1.13524913e+02, 0.00000000e+00, 5.56421371e-02],
                     [1.26356287e+02, 0.00000000e+00, 5.48046877e-02],
                     [4.17379637e+01, 0.00000000e+00, 1.20566608e-03],
                     [2.87871488e+03, 0.00000000e+00, 8.19146145e+00],
                     [2.88337138e+03, 0.00000000e+00, 8.24032840e+00],
                     [9.10654044e+02, 0.00000000e+00, 8.26470640e+00],
                     [5.26836236e+02, 0.00000000e+00, 1.97381003e+01],
                     [9.18064763e+03, 0.00000000e+00, 1.97215311e+01],
                     [3.78463739e+03, 0.00000000e+00, 1.96914248e+01],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

# Training data
dmax = np.array([2.5664000e-02, 4.1890000e-02, 0.0000000e+00,  # cart pos
                 4.2490000e-02, 4.4770000e-02, 0.0000000e+00,  # cart vel
                 1.9012000e-02, 7.1500000e-03, 8.4417000e-02, 2.3041000e-02,  # pole quat
                 2.8452940e+00, 3.3939100e+00, 1.3531405e+01,  # pole vel
                 4.4438300e-01, 4.3333600e-01, 4.1041900e-01,  # block pos
                 2.22191570e-01, 2.26668060e-01, 2.05209520e-01])  # block vel Max  .. TB adjusted by had given how many errors in normal runs for phase 3 code..
#                                     1.4438300e-02, 1.3333600e-01, 1.1041900e-01, #block pos
#                                     1.9191570e-01, 1.96668060e-01, 1.95209520e-01]) #block vel Max

diffwbl = np.array([[[3.89207845e-01, 0.00000000e+00, 1.26112630e-02],
                     [2.08246398e+01, 0.00000000e+00, 8.14071727e-02],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [1.80495509e+00, 0.00000000e+00,
                      5.13721453e-01],  # TB hand adjusted
                     [1.39238147e+00, 0.00000000e+00,
                      1.63304631e-01],  # TB hand adjusted
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [2.39054403e+00, 0.00000000e+00, 6.48172095e-03],
                     [8.41732321e-01, 0.00000000e+00, 3.70890868e-03],
                     [1.46458996e+00, 0.00000000e+00, 2.27123147e-02],
                     [1.48986272e+00, 0.00000000e+00, 3.98654470e-02],
                     [4.55443947e+00, 0.00000000e+00, 1.54452261e+00],
                     [5.44053827e+00, 0.00000000e+00, 9.21315026e-01],
                     [3.23239311e+00, 0.00000000e+00, 1.13444676e+00],
                     [4.68533570e+00, 0.00000000e+00, 1.45315737e-02],
                     [1.20077120e+00, 0.00000000e+00, 1.51368367e-02],
                     [2.31710759e+00, 0.00000000e+00, 2.05984072e-02],
                     [4.68573228e+00, 0.00000000e+00, 7.26578269e-01],
                     [1.20072688e+00, 0.00000000e+00, 7.56841161e-01],
                     [2.31709025e+00, 0.00000000e+00, 1.02992254e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                    [[1.35507369e+01, 0.00000000e+00, 5.81425563e-02],
                     [8.43288360e+02, 0.00000000e+00, 8.20148859e-02],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [9.21130389e-01, 0.00000000e+00, 4.34356316e-03],
                     [1.42393753e+00, 0.00000000e+00, 8.87714776e-03],
                     [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                     [5.64946247e+00, 0.00000000e+00, 5.85444651e-02],
                     [1.11560747e+01, 0.00000000e+00, 2.70439053e-02],
                     [9.89439561e+00, 0.00000000e+00, 1.63992817e-01],
                     [2.69850797e+01, 0.00000000e+00, 1.51578428e-01],
                     [1.03975272e+02, 0.00000000e+00, 3.13070568e+00],
                     [2.80463669e+01, 0.00000000e+00, 3.75921232e+00],
                     [1.54359124e+01, 0.00000000e+00, 2.21769648e+01],
                     [2.18464037e+01, 0.00000000e+00, 3.00350669e-01],
                     [8.55836582e+01, 0.00000000e+00, 2.92873593e-01],
                     [3.52209971e+01, 0.00000000e+00, 2.62699935e-01],
                     [2.18463402e+01, 0.00000000e+00, 1.50175196e+01],
                     [8.55804857e+01, 0.00000000e+00, 1.46436707e+01],
                     [3.52198475e+01, 0.00000000e+00, 1.31349703e+01],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

# abuse tha terms since we are tryng to share code with  EVT version
imean = np.array([-3.88847920e-02,  1.79388544e-02,  0.00000000e+00, -8.32127574e-05,
                  5.61661668e-05,  0.00000000e+00, -1.52076385e-04, -1.22921416e-05,
                  9.18328334e-05,  9.99950191e-01, -2.47030594e-05,  2.25482903e-04,
                  3.81943611e-06, -3.34521356e-03, -3.63842781e-02,  5.03422818e+00,
                  -1.33613935e-02, -3.96377125e-02,  3.47097780e-03, -5.00000000e+00,
                  -5.00000000e+00,  0.00000000e+00,  5.00000000e+00, -5.00000000e+00,
                  0.00000000e+00,  5.00000000e+00,  5.00000000e+00,  0.00000000e+00,
                  -5.00000000e+00,  5.00000000e+00,  0.00000000e+00, -5.00000000e+00,
                  -5.00000000e+00,  1.00000000e+01,  5.00000000e+00, -5.00000000e+00,
                  1.00000000e+01,  5.00000000e+00,  5.00000000e+00,  1.00000000e+01,
                  -5.00000000e+00,  5.00000000e+00,  1.00000000e+01])

# adjusted based on code..
istd = np.array([1.70842105e+00, 1.72660247e+00, 1.00000000e-06, 1.14153952e-02,
                 1.14772434e-02, 1.00000000e-06, 5.79354090e-03, 5.75740406e-03,
                 5.73292428e-03, 2.60735246e-05, 1.21789231e-02, 1.21918190e-02,
                 1.96198184e-04, 2.32131483e+00, 2.33097704e+00, 2.29533051e+00,
                 7.97934966e+00, 7.97332509e+00, 8.02280604e+00, 1.00000000e-06,
                 1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                 1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                 1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                 1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                 1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                 1.00000000e-06, 1.00000000e-06, 1.00000000e-06])

# load data from triningn
dmean = np.array([-3.13321142e-07,  2.76174424e-05,  0.00000000e+00,
                  -3.22495249e-05, -4.54427372e-05,  0.00000000e+00,
                  -1.96683762e-06,  4.17344353e-06, -3.58822868e-06,  1.24106557e-04,
                  -1.90599960e-03,  6.85632951e-04, 2.24843167e-03,
                  -2.08509736e-06, -1.15790544e-05,  9.71314184e-05,
                  -1.04261269e-04, -5.78904334e-04,  4.85657617e-03])
dstd = np.array([-3.13321142e-07,  2.76174424e-05,  1.00000000e-06,
                 -3.22495249e-05, -4.54427372e-05,  1.00000000e-06,
                 -1.96683762e-06,  4.17344353e-06, -3.58822868e-06,  1.24106557e-04,
                 -1.90599960e-03,  6.85632951e-04, 2.24843167e-03,
                 -2.08509736e-06, -1.15790544e-05,  9.71314184e-05,
                 -1.04261269e-04, -5.78904334e-04,  4.85657617e-03])

cntwbl = np.array([[1.44423424, -0.19000001,  0.14582359],
                   [0.77655005, -0.13000001,  0.07412372],
                   [1.4183835, -0.08000001,  0.06339877],
                   [1.3369761, -0.36100001,  0.04456476],
                   [0.84213063, -0.37600001,  0.03392788],
                   [0.84213063, -0.36500001,  0.03392788],
                   [1.60734717, -0.35000001,  0.07514313],
                   [1.37548511, -0.26000001,  0.11283754],
                   [1.18602747, -0.21000001,  0.0627653],
                   [1.70217044, -0.06000001,  0.05617951],
                   [1.50681693, -0.06000001,  0.05125134],
                   [1.50681693, -0.06000001,  0.05125134],
                   [0.60246664, -0.08000001,  0.0598179]
                   ])

avoid_list = [
    #            ['left','left','left','left','nothing','nothing','nothing','nothing','right','right','right'],
    #            ['left','left','left','nothing','nothing','nothing','right','right'],
    #            ['right','right','right','nothing','nothing','nothing','left','left',],
    #            ['forward','forward','forward','nothing','nothing','nothing','backward','backward'],
    #            ['backward','backward','backward','nothing','nothing','nothing','forward','forward']
    ['left', 'left', 'left', 'nothing', 'nothing', 'nothing'],
    ['right', 'right', 'right', 'nothing', 'nothing', 'nothing'],
    ['forward', 'forward', 'forward', 'nothing', 'nothing', 'nothing'],
    ['backward', 'backward', 'backward', 'nothing', 'nothing', 'nothing']
]

actions_plist = [(0, 1, 2, 3, 4),  # normal
                 # swap left/right (lave front/back)      keep major dim order
                 (0, 2, 1, 3, 4),
                 # swap  swap left right and  front/back  keep major dim order
                 (0, 2, 1, 4, 3),
                 # swap keep major dim order front/back items with left right leaving minor dim ordering
                 (0, 3, 4, 1, 2),
                 (0, 4, 3, 2, 1),  # swap  major and minor
                 # rest are just remaining pertubations in  standard pertubation order.
                 (0, 1, 2, 4, 3), (0, 1, 3, 2, 4), (0, 1, 3,
                                                    4, 2), (0, 1, 4, 2, 3), (0, 1, 4, 3, 2),
                 (0, 2, 3, 1, 4), (0, 2, 3, 4, 1), (0, 2, 4,
                                                        1, 3), (0, 2, 4, 3, 1), (0, 3, 1, 2, 4),
                 (0, 3, 1, 4, 2), (0, 3, 2, 1, 4), (0, 3, 2,
                                                        4, 1),  (0, 3, 4, 2, 1),  (0, 4, 1, 2, 3),
                 (0, 4, 1, 3, 2), (0, 4, 2, 1, 3), (0,
                                                        4, 2, 3, 1), (0, 4, 3, 1, 2),
                 (1, 0, 2, 3, 4), (1, 0, 2, 4, 3), (1, 0, 3,
                                                        2, 4), (1, 0, 3, 4, 2), (1, 0, 4, 2, 3),
                 (1, 0, 4, 3, 2), (1, 2, 0, 3, 4), (1, 2, 0,
                                                        4, 3), (1, 2, 3, 0, 4), (1, 2, 3, 4, 0),
                 (1, 2, 4, 0, 3), (1, 2, 4, 3, 0), (1, 3, 0,
                                                        2, 4), (1, 3, 0, 4, 2), (1, 3, 2, 0, 4),
                 (1, 3, 2, 4, 0), (1, 3, 4, 0, 2), (1, 3, 4,
                                                        2, 0), (1, 4, 0, 2, 3), (1, 4, 0, 3, 2),
                 (1, 4, 2, 0, 3), (1, 4, 2, 3, 0), (1,
                                                        4, 3, 0, 2), (1, 4, 3, 2, 0),
                 (2, 0, 1, 3, 4), (2, 0, 1, 4, 3), (2, 0, 3,
                                                        1, 4), (2, 0, 3, 4, 1), (2, 0, 4, 1, 3),
                 (2, 0, 4, 3, 1), (2, 1, 0, 3, 4), (2, 1, 0,
                                                        4, 3), (2, 1, 3, 0, 4), (2, 1, 3, 4, 0),
                 (2, 1, 4, 0, 3), (2, 1, 4, 3, 0), (2, 3, 0,
                                                        1, 4), (2, 3, 0, 4, 1), (2, 3, 1, 0, 4),
                 (2, 3, 1, 4, 0), (2, 3, 4, 0, 1), (2, 3, 4,
                                                        1, 0), (2, 4, 0, 1, 3), (2, 4, 0, 3, 1),
                 (2, 4, 1, 0, 3), (2, 4, 1, 3, 0), (2,
                                                        4, 3, 0, 1), (2, 4, 3, 1, 0),
                 (3, 0, 1, 2, 4), (3, 0, 1, 4, 2), (3, 0, 2,
                                                        1, 4), (3, 0, 2, 4, 1), (3, 0, 4, 1, 2),
                 (3, 0, 4, 2, 1), (3, 1, 0, 2, 4), (3, 1, 0,
                                                        4, 2), (3, 1, 2, 0, 4), (3, 1, 2, 4, 0),
                 (3, 1, 4, 0, 2), (3, 1, 4, 2, 0), (3, 2, 0,
                                                        1, 4), (3, 2, 0, 4, 1), (3, 2, 1, 0, 4),
                 (3, 2, 1, 4, 0), (3, 2, 4, 0, 1), (3, 2, 4,
                                                        1, 0), (3, 4, 0, 1, 2), (3, 4, 0, 2, 1),
                 (3, 4, 1, 0, 2), (3, 4, 1, 2, 0), (3, 4, 2,
                                                        0, 1), (3, 4, 2, 1, 0), (4, 0, 1, 2, 3),
                 (4, 0, 1, 3, 2), (4, 0, 2, 1, 3), (4, 0, 2,
                                                        3, 1), (4, 0, 3, 1, 2), (4, 0, 3, 2, 1),
                 (4, 1, 0, 2, 3), (4, 1, 0, 3, 2), (4, 1, 2,
                                                        0, 3), (4, 1, 2, 3, 0), (4, 1, 3, 0, 2),
                 (4, 1, 3, 2, 0), (4, 2, 0, 1, 3), (4, 2, 0,
                                                        3, 1), (4, 2, 1, 0, 3), (4, 2, 1, 3, 0),
                 (4, 2, 3, 0, 1), (4, 2, 3, 1, 0), (4, 3, 0,
                                                        1, 2), (4, 3, 0, 2, 1), (4, 3, 1, 0, 2),
                 (4, 3, 1, 2, 0), (4, 3, 2, 0, 1), (4, 3, 2, 1, 0)]
