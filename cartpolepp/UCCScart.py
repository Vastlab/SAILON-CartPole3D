"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/book/code/pole.c
"""

import os
import sys
import time
import math
import pdb

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import pybullet as p2
from pybullet_utils import bullet_client as bc


class CartPoleBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, params: dict = None):
        # start the bullet physics server
        self._render_height = 480
        self._render_width = 640
        self._physics_client_id = -1
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 0.4  # 2.4
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2, np.finfo(np.float32).max])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Environmental params
        self.force_mag = 10
        self.timeStep = 1.0 / 50.0
        self.angle_limit = 10.0 * np.pi / 180.0 # 10 degrees in radians
        self.actions = ['right', 'left', 'forward', 'backward', 'nothing']
        self.tick_limit = 200

        # Internal params
        self.params = params
        self.path = self.params['path']
        self._renders = self.params['use_gui']
        self.tick = 0
        self.time = None
        self.np_random = None
        self.use_img = self.params['use_img']
        self._p = None

        # Params
        self.init_zero = False
        self.init_zero = True        
        self.config = self.params['config']

        # Object definitions
        self.nb_blocks = None
        self.cartpole = -10
        self.ground = None
        self.blocks = list()
        self.walls = None
        self.state = None
        self.origin = None

        # Functions to be run directly after init
        self.seed(self.params['seed'])

        #TB additions/hacks
        self.setbase=False
        self.basepos=[0,0,0]
        self.lastscore=0
        self.givendetection=0        
        self.wcprob=0
        self.char=""
        self.tbdebuglevel=0
        self.episode=0
        self.force_action=-1
        self.runandhide=0    # how much weight to we put on running and hiding.  If 1 we will hide in corner, if <.1  we ignore collistions and  < .5  we increase weight collisions up to 1 and above that we increase weight in to hiding in corner 

        self.use_avoid_reaction=False
        self.reactstep=0
        self.avoid_list = [
#            ['left','left','left','left','nothing','nothing','nothing','nothing','right','right','right'],
            ['left','left','left','nothing','nothing','nothing','right','right'],            
            ['right','right','right','nothing','nothing','nothing','left','left',],
            ['forward','forward','forward','nothing','nothing','nothing','backward','backward'],
            ['backward','backward','backward','nothing','nothing','nothing','forward','forward']                                                    
        ]
        self.avoid_actions= self.avoid_list[0]


        # where we story action/state history.. first dimension is action number, second is action (0-4 for stated from actions 0-4), 5 is action choice (only first value used), 6 is expected state, 7 is actual state returned,
        # note we use format_data to reduce dictionary to a numeeric vector for faster comparisons but we ignore

        self.action_history = np.zeros((5,2,13))
        
        self.actions_permutation_index=0
        self.actions_plist= [(0, 1, 2, 3, 4),  #normal
                             (0, 2, 1, 3, 4), #swap left/right (lave front/back)      keep major dim order
                             (0, 2, 1, 4, 3),  #swap  swap left right and  front/back  keep major dim order
                             (0, 3, 4, 1, 2),  #swap keep major dim order front/back items with left right leaving minor dim ordering
                             (0, 4, 3, 2, 1),  #swap  major and minor                            
                             #rest are just remaining pertubations in  standard pertubation order. 
                             (0, 1, 2, 4, 3), (0, 1, 3, 2, 4), (0, 1, 3, 4, 2), (0, 1, 4, 2, 3), (0, 1, 4, 3, 2),
                             (0, 2, 3, 1, 4), (0, 2, 3, 4, 1), (0, 2, 4, 1, 3), (0, 2, 4, 3, 1), (0, 3, 1, 2, 4),
                             (0, 3, 1, 4, 2), (0, 3, 2, 1, 4), (0, 3, 2, 4, 1),  (0, 3, 4, 2, 1),  (0, 4, 1, 2, 3),
                             (0, 4, 1, 3, 2), (0, 4, 2, 1, 3), (0, 4, 2, 3, 1), (0, 4, 3, 1, 2),
                             (1, 0, 2, 3, 4), (1, 0, 2, 4, 3), (1, 0, 3, 2, 4), (1, 0, 3, 4, 2), (1, 0, 4, 2, 3),
                             (1, 0, 4, 3, 2), (1, 2, 0, 3, 4), (1, 2, 0, 4, 3), (1, 2, 3, 0, 4), (1, 2, 3, 4, 0),
                             (1, 2, 4, 0, 3), (1, 2, 4, 3, 0), (1, 3, 0, 2, 4), (1, 3, 0, 4, 2), (1, 3, 2, 0, 4),
                             (1, 3, 2, 4, 0), (1, 3, 4, 0, 2), (1, 3, 4, 2, 0), (1, 4, 0, 2, 3), (1, 4, 0, 3, 2),
                             (1, 4, 2, 0, 3), (1, 4, 2, 3, 0), (1, 4, 3, 0, 2), (1, 4, 3, 2, 0),
                             (2, 0, 1, 3, 4), (2, 0, 1, 4, 3), (2, 0, 3, 1, 4), (2, 0, 3, 4, 1), (2, 0, 4, 1, 3),
                             (2, 0, 4, 3, 1), (2, 1, 0, 3, 4), (2, 1, 0, 4, 3), (2, 1, 3, 0, 4), (2, 1, 3, 4, 0),
                             (2, 1, 4, 0, 3), (2, 1, 4, 3, 0), (2, 3, 0, 1, 4), (2, 3, 0, 4, 1), (2, 3, 1, 0, 4),
                             (2, 3, 1, 4, 0), (2, 3, 4, 0, 1), (2, 3, 4, 1, 0), (2, 4, 0, 1, 3), (2, 4, 0, 3, 1),
                             (2, 4, 1, 0, 3), (2, 4, 1, 3, 0), (2, 4, 3, 0, 1), (2, 4, 3, 1, 0),
                             (3, 0, 1, 2, 4), (3, 0, 1, 4, 2), (3, 0, 2, 1, 4), (3, 0, 2, 4, 1), (3, 0, 4, 1, 2),
                             (3, 0, 4, 2, 1), (3, 1, 0, 2, 4), (3, 1, 0, 4, 2), (3, 1, 2, 0, 4), (3, 1, 2, 4, 0),
                             (3, 1, 4, 0, 2), (3, 1, 4, 2, 0), (3, 2, 0, 1, 4), (3, 2, 0, 4, 1), (3, 2, 1, 0, 4),
                             (3, 2, 1, 4, 0), (3, 2, 4, 0, 1), (3, 2, 4, 1, 0), (3, 4, 0, 1, 2), (3, 4, 0, 2, 1),
                             (3, 4, 1, 0, 2), (3, 4, 1, 2, 0), (3, 4, 2, 0, 1), (3, 4, 2, 1, 0), (4, 0, 1, 2, 3),
                             (4, 0, 1, 3, 2), (4, 0, 2, 1, 3), (4, 0, 2, 3, 1), (4, 0, 3, 1, 2), (4, 0, 3, 2, 1),
                             (4, 1, 0, 2, 3), (4, 1, 0, 3, 2), (4, 1, 2, 0, 3), (4, 1, 2, 3, 0), (4, 1, 3, 0, 2),
                             (4, 1, 3, 2, 0), (4, 2, 0, 1, 3), (4, 2, 0, 3, 1), (4, 2, 1, 0, 3), (4, 2, 1, 3, 0),
                             (4, 2, 3, 0, 1), (4, 2, 3, 1, 0), (4, 3, 0, 1, 2), (4, 3, 0, 2, 1), (4, 3, 1, 0, 2),
                             (4, 3, 1, 2, 0), (4, 3, 2, 0, 1), (4, 3, 2, 1, 0)]
        self.actions_permutation_tried=np.zeros(len(self.actions_plist))
        self.actions_permutation_tried[0] = 1
        

        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return None

    def string_to_actionnum(self,action):
        # Convert from string to int
        if action == 'nothing' or action ==0:
            action = 0
        elif action == 'right':
            action = 1    
        elif action == 'left':
            action = 2
        elif action == 'forward':
            action = 3
        elif action == 'backward':
            action = 4
        # else ifits a number leave it alone, but warn
        else:
            if(not (type(action)==  int and action >=0 and action <= 4)):
                print("invalid action", action)
            #else its a number so just use it..   reset call here with 
        return action

    def actionnum_to_string(self,action):
        #Convert from string to int
        if action == 0: action = 'nothing'
        elif action == 1:action = 'right'
        elif action == 2: action = 'left'
        elif action == 3: action = 'forward'
        elif action == 4: action = 'backward'
        # else ifits a number leave it alone, but warn
        else: print("invalid action", action)
        
        return action


    def step(self, action):
        p = self._p
        
        action = self.string_to_actionnum(action)        

        #apply TB's preturbation search as needed (something remote may change actions_permutation_index to drive the search.. here we jsut use current index)
        action = self.actions_plist[self.actions_permutation_index][action]


        # Adjust forces so they always apply in reference to world frame
        _, ori, _, _, _, _ = p.getLinkState(self.cartpole, 0)
        cart_angle = p.getEulerFromQuaternion(ori)[2] # yaw
        fx = self.force_mag * np.cos(cart_angle)
        fy = self.force_mag * np.sin(cart_angle) * -1

        # based on action decide the x and y forces
        if action == 0:
            fx = 0.0
            fy = 0.0
        elif action == 1:
            fx = fx
            fy = fy
        elif action == 2:
            fx = -fx
            fy = - fy
        elif action == 3:
            tmp = fx
            fx = -fy
            fy = tmp
        elif action == 4:
            tmp = fx
            fx = fy
            fy = -tmp
        else:
            raise Exception("unknown discrete action [%s]" % action)

        # Apply correccted forces
        p.applyExternalForce(self.cartpole, 0, (fx, fy, 0.0), (0, 0, 0), p.LINK_FRAME)

        # Apply anti-gravity to blocks
        for i in self.blocks:
            p.applyExternalForce(i, -1, (0, 0, 9.8), (0, 0, 0), p.LINK_FRAME)

        p.stepSimulation()

        done = self.is_done()
        reward = self.get_reward()

#tb remove tick here since not updated inreset..         
#        self.tick = self.tick + 1

        return self.get_state(), reward, done, {}

    # Check if is done
    def is_done(self):
        # Check tick limit condition
        if self.tick >= self.tick_limit:
            return True

        # Check pole angle condition
        p = self._p
        _, _, _, _, _, ori, _, _ = p.getLinkState(self.cartpole, 1, 1)
        eulers = p.getEulerFromQuaternion(ori)
        x_angle, y_angle = eulers[0], eulers[1]

        if abs(x_angle) > self.angle_limit or abs(y_angle) > self.angle_limit:
            return True
        else:
            return False

        return None

    def get_reward(self):
        return self.tick / self.tick_limit

    def get_time(self):
        return self.time + self.tick * self.timeStep

    def get_actions(self):
        return self.actions

    def resetbase(self):
        self.setbase=True
        return None


    def reset(self, feature_vector=None):
        # Set time paremeter for sensor value
        self.time = time.time()

        # Create client if it doesnt exist
        if self._physics_client_id < 0:
            self.generate_world()

        self.reset_world()

        # Run for one step to get everything going
        if(feature_vector is None):
            self.tick = 0     # if not a reset to given state  we reset tick
            self.episode=0            
            self.step('nothing')
            self.reactstep=0            
            self.force_action=-1
        else:
            self.set_world(feature_vector)

        return self.get_state(initial=True)

    # Used to generate the initial world state
    def generate_world(self):
        # Read user config here
        if self.config is not None:
            if 'start_zeroed_out' in self.config:
                self.init_zero = self.config['start_zeroed_out']
            if 'episode_seed' in self.config:
                self.seed(self.config['episode_seed'])
            if 'start_world_state' in self.config:
                self.set_world(self.config['start_world_state'])

        # Create bullet physics client
        if self._renders:
            self._p = bc.BulletClient(connection_mode=p2.GUI)
        else:
            self._p = bc.BulletClient(connection_mode=p2.DIRECT)
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K") # Clear to the end of line

        # Client id link, for closing or checking if running
        self._physics_client_id = self._p._client

        # Load world simulation
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        # Load world objects
        self.cartpole = p.loadURDF(os.path.join(self.path, 'models', 'ground_cart.urdf'))
        self.walls = p.loadURDF(os.path.join(self.path, 'models', 'walls.urdf'))
        self.origin = p.loadURDF(os.path.join(self.path, 'models', 'origin.urdf'))

        # Set walls to be bouncy
        for joint_nb in range(-1, 6):
            p.changeDynamics(self.walls, joint_nb, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        return None

    def reset_world(self):
        # Reset world (assume is created)
        p = self._p

        # Delete cartpole
        if self.cartpole == -10:
            self.cartpole = p.loadURDF(os.path.join(self.path, 'models', 'ground_cart.urdf'))
        else:
            p.removeBody(self.cartpole)
            self.cartpole = p.loadURDF(os.path.join(self.path, 'models', 'ground_cart.urdf'))

        # This big line sets the spehrical joint on the pole to loose
        p.setJointMotorControlMultiDof(self.cartpole, 1, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1],
                                       targetVelocity=[0, 0, 0], positionGain=0, velocityGain=0.0,
                                       force=[0, 0, 0])

        # Reset cart (technicaly ground object)
        if self.init_zero:
            cart_pos = list(self.np_random.uniform(low=0, high=0, size=(2,))) + [0]
            cart_vel = list(self.np_random.uniform(low=0, high=0, size=(2,))) + [0]
        else:
            cart_pos = list(self.np_random.uniform(low=-3, high=3, size=(2,))) + [0]
            cart_vel = list(self.np_random.uniform(low=-1, high=1, size=(2,))) + [0]

        p.resetBasePositionAndOrientation(self.cartpole, cart_pos, [0, 0, 0, 1])
        p.applyExternalForce(self.cartpole, 0, cart_vel, (0, 0, 0), p.LINK_FRAME)

        # Reset pole
        if self.init_zero:
            randstate = list(self.np_random.uniform(low=0, high=0, size=(6,)))
        else:
            randstate = list(self.np_random.uniform(low=-0.01, high=0.01, size=(6,)))

        pole_pos = randstate[0:3] + [1]
        # zero so it doesnt spin like a top :)
        pole_ori = list(randstate[3:5]) + [0]
        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_pos, targetVelocity=pole_ori)

        # Delete old blocks
        for i in self.blocks:
            p.removeBody(i)

        # Load blocks in
        self.nb_blocks = np.random.randint(3) + 2
        if(self.init_zero): self.nb_blocks=0        
        self.blocks = [None] * self.nb_blocks
        for i in range(self.nb_blocks):
            self.blocks[i] = p.loadURDF(os.path.join(self.path, 'models', 'block.urdf'))

        # Set blocks to be bouncy
        for i in self.blocks:
            p.changeDynamics(i, -1, restitution=1.0, lateralFriction=0.0,
                             rollingFriction=0.0, spinningFriction=0.0)

        # Set block posistions
        min_dist = 1
        cart_pos, _ = p.getBasePositionAndOrientation(self.cartpole)
        cart_pos = np.asarray(cart_pos)
        for i in self.blocks:
            pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
            pos[2] = pos[2] + 5.0
            while np.linalg.norm(cart_pos[0:2] - pos[0:2]) < min_dist:
                pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))
                # Z is not centered at 0.0
                pos[2] = pos[2] + 5.0
            p.resetBasePositionAndOrientation(i, pos, [0, 0, 1, 0])

        # Set block velocities
        for i in self.blocks:
            vel = self.np_random.uniform(low=6.0, high=10.0, size=(3,))
            for ind, val in enumerate(vel):
                if np.random.rand() < 0.5:
                    vel[ind] = val * -1

            p.resetBaseVelocity(i, vel, [0, 0, 0])

        p.stepSimulation()            

        return None

    def set_world(self, state):
#        print('TBs Set World only approximately implemented :(')
        p = self._p        


        cart_position = [state["cart"]["x_position"],  state["cart"]["y_position"],state["cart"]["z_position"]]
        # we swap x and z for velocity interface to pybullet        
        cart_velocity = [state["cart"]["z_velocity"], state["cart"]["y_velocity"],state["cart"]["x_velocity"]]
        
        p.resetBasePositionAndOrientation(self.cartpole, cart_position, [0, 0, 0, 1])
        p.resetJointStateMultiDof(self.cartpole, 0, targetValue=[0,0,0], targetVelocity=cart_velocity)


        # Reset pole
        pole_position = [state["pole"]["x_quaternion"],state["pole"]["y_quaternion"],state["pole"]["z_quaternion"],state["pole"]["w_quaternion"]]
        pole_velocity = [state["pole"]["x_velocity"],state["pole"]["y_velocity"],state["pole"]["z_velocity"]*0]
        p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_position, targetVelocity=pole_velocity)

        # Delete old blocks if number is different
        if(len(state['blocks']) != self.nb_blocks):
            for i in self.blocks:
                p.removeBody(i)

            self.nb_blocks = len(state['blocks'])
            if(self.nb_blocks > 4):
                self.character += "& Too many blocks" +str(self.nb_blocks)
                self.wcprob=1               
            elif(self.nb_blocks < 2):
                self.character += "& Too few blocks" +str(self.nb_blocks)                
                self.wcprob=1                               
                
            self.blocks = [None] * self.nb_blocks
            for i in range(self.nb_blocks):
                self.blocks[i] = p.loadURDF(os.path.join(self.path, 'models', 'block.urdf'))
                    
        i=0
        for block in state["blocks"]:
            pos = [block["x_position"], block["y_position"],block["z_position"]]
            vel = [block["x_velocity"], block["y_velocity"],block["z_velocity"]]
            p.resetBasePositionAndOrientation(self.blocks[i], pos, [0, 0, 1, 0])
            p.resetBaseVelocity(self.blocks[i], vel, [0, 0, 0])
            i = i+1

#        p.stepSimulation()            

        return None

    
#     def set_state(self, state):
#         print('Set World is not yet fully tested :(')
#         p = self._p        
#         cart_position = [state["cart"]["x_position"],   # x if flipped compard to pybullet
#                          state["cart"]["y_position"],
#                          state["cart"]["z_position"]
#         ]
#         cart_velocity = [state["cart"]["z_velocity"],    # we swap x and z for interface to pybullet
#                          state["cart"]["y_velocity"],
#                          state["cart"]["x_velocity"]
#         ]
        
#         base_positione, _ = p.getBasePositionAndOrientation(self.cartpole)
#         cartoffset = [0,0,0]
#         self.setbase=True        
#         if(self.setbase):
#             self.setbase=False
#             self.basepos=cart_position
#             # if base is 0 then set meaningful base from features
#             #                print("reset car base from ", base_positione, " to ",  cart_position)
#             #                cart_position = [.80,.80,0]
#             p.resetBasePositionAndOrientation(self.cartpole, cart_position, [0, 0, 0, 1])
#             p.resetJointStateMultiDof(self.cartpole, 0, targetValue=[0,0,0], targetVelocity=cart_velocity)
# #            p.resetBasePositionAndOrientation(self.cartpole, [0,0,0], [0, 0, 0, 1])            
# #            p.resetJointStateMultiDof(self.cartpole, 0, targetValue=cart_position, targetVelocity=cart_velocity)

#             _, _, _, _, _, _, vel, _ = p.getLinkState(self.cartpole, 0, 1)
#             pos, _, _, _, _, _ = p.getLinkState(self.cartpole, 0)
#  #           print("Setbase  reset  from ", cart_position,cart_velocity, " to ",  pos,vel)
#         else:
#             #use two part model with base and link  .. ut so far not working right might need a bullet library fix
#             base_position = self.basepos
#             cart_offset = [state["cart"]["x_position"]-base_position[0],state["cart"]["y_position"]-base_position[1],
#                           (state["cart"]["z_position"]-base_position[2]) 
#             ]
#             p.resetBasePositionAndOrientation(self.cartpole, base_position, [0, 0, 0, 1])                    
#             p.resetJointStateMultiDof(self.cartpole, 0, targetValue=cart_offset,targetVelocity=cart_velocity)
      
#             _, vel, _, _ = p.getJointStateMultiDof(self.cartpole, 0)
#             pos, _, _, _, _, _ = p.getLinkState(self.cartpole, 0)
#             print("Normal reset  from ", cart_position,cart_velocity, "with base/offset", base_position, cart_offset,  " to ",  pos,vel)

#         # Reset pole
#         pole_position = [state["pole"]["x_quaternion"],state["pole"]["y_quaternion"],state["pole"]["z_quaternion"],state["pole"]["w_quaternion"]]
#         pole_velocity = [state["pole"]["x_velocity"],state["pole"]["y_velocity"],state["pole"]["z_velocity"]*0]
#         p.resetJointStateMultiDof(self.cartpole, 1, targetValue=pole_position, targetVelocity=pole_velocity)

#         # Should really check if number change as that could be novelty but for now just copy them
#         # Delete old blocks
#         if(len(state['blocks']) != self.nb_blocks):
#             for i in self.blocks:
#                 p.removeBody(i)

#             self.nb_blocks = len(state['blocks'])
#             self.blocks = [None] * self.nb_blocks
#             for i in range(self.nb_blocks):
#                 self.blocks[i] = p.loadURDF(os.path.join(self.path, 'models', 'block.urdf'))
                    
#         i=0
#         for block in state["blocks"]:
#             pos = [block["x_position"],block["y_position"],block["z_position"]]
#             vel = [block["x_velocity"],block["y_velocity"],block["z_velocity"]]
#             p.resetBasePositionAndOrientation(self.blocks[i], pos, [0, 0, 1, 0])
#             p.resetBaseVelocity(self.blocks[i], vel, [0, 0, 0])
#             i = i+1

        
#         return None

    # Unified function for getting state information
    def get_state(self, initial=False):
        p = self._p
        world_state = dict()
        round_amount = 6

        # Get cart info ============================================
        state = dict()

        # Handle pos, vel
        pos, _, _, _, _, _ = p.getLinkState(self.cartpole, 0)
        state['x_position'] = round(pos[0], round_amount)
        state['y_position'] = round(pos[1], round_amount)
        state['z_position'] = round(pos[2], round_amount)

        # Cart velocity from planar joint (buggy in PyBullet; thus reverse order)
        # _, vel, _, _ = p.getJointStateMultiDof(self.cartpole, 0)
        # state['x_velocity'] = round(vel[2], round_amount)
        # state['y_velocity'] = round(vel[1], round_amount)
        # state['z_velocity'] = round(vel[0], round_amount)

        # Cart velocity from cart
        _, _, _, _, _, _, vel, _ = p.getLinkState(self.cartpole, 0, 1)
        state['x_velocity'] = round(vel[0], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        state['z_velocity'] = round(vel[2], round_amount)

        # Set world state of cart
        world_state['cart'] = state

        # Get pole info =============================================
        state = dict()
        use_euler = False

        # Orientation and A_velocity, the others not used
        _, _, _, _, _, ori, _, vel = p.getLinkState(self.cartpole, 1, 1)

        # Orientation
        if use_euler:
            # Convert quats to eulers
            eulers = p.getEulerFromQuaternion(ori)
            state['x_euler'] = round(eulers[0], round_amount)
            state['y_euler'] = round(eulers[1], round_amount)
            state['z_euler'] = round(eulers[2], round_amount)
        else:
            state['x_quaternion'] = round(ori[0], round_amount)
            state['y_quaternion'] = round(ori[1], round_amount)
            state['z_quaternion'] = round(ori[2], round_amount)
            state['w_quaternion'] = round(ori[3], round_amount)

        # A_velocity
        state['x_velocity'] = round(vel[0], round_amount)
        state['y_velocity'] = round(vel[1], round_amount)
        state['z_velocity'] = round(vel[2], round_amount)

        world_state['pole'] = state

        # get block info ====================================
        block_state = list()
        for ind, val in enumerate(self.blocks):
            state = dict()
            state['id'] = val

            pos, _ = p.getBasePositionAndOrientation(val)
            state['x_position'] = round(pos[0], round_amount)
            state['y_position'] = round(pos[1], round_amount)
            state['z_position'] = round(pos[2], round_amount)

            vel, _ = p.getBaseVelocity(val)
            state['x_velocity'] = round(vel[0], round_amount)
            state['y_velocity'] = round(vel[1], round_amount)
            state['z_velocity'] = round(vel[2], round_amount)

            block_state.append(state)

        world_state['blocks'] = block_state

        # Get wall info ======================================
        if initial:
            state = list()
            state.append([-5, -5, 0])
            state.append([5, -5, 0])
            state.append([5, 5, 0])
            state.append([-5, 5, 0])

            state.append([-5, -5, 10])
            state.append([5, -5, 10])
            state.append([5, 5, 10])
            state.append([-5, 5, 10])

            world_state['walls'] = state

        return world_state

    def get_image(self):
        if self.use_img:
            return self.render()
        else:
            return None

    def render(self, mode='human', close=False, dist='close'):
        if mode == "human":
            self._renders = True

        if dist == 'far':
            base_pos = [4.45, 4.45, 9.8]
            cam_dist = 0.1
            cam_pitch = -45.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 100

        elif dist == 'close':
            base_pos = [4.45, 4.45, 2.0]
            cam_dist = 0.1
            cam_pitch = -15.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 60

        elif dist == 'follow':
            base_pose, _ = self._p.getBasePositionAndOrientation(self.cartpole)
            pos, vel, jRF, aJMT = self._p.getJointStateMultiDof(self.cartpole, 0)

            x = pos[0] + base_pose[0]
            y = pos[1] + base_pose[1]

            base_pos = [x, y, 2.0]
            cam_dist = 0.1
            cam_pitch = -15.0
            cam_yaw = 45.0 + 90
            cam_roll = 0.0
            fov = 60

        if self._physics_client_id >= 0:
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=cam_dist,
                yaw=cam_yaw,
                pitch=cam_pitch,
                roll=cam_roll,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=fov,
                                                             aspect=float(self._render_width) /
                                                                    self._render_height,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
        else:
            px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def format_data(self, feature_vector):
        # Format data for use with evm

        state = []
        for i in feature_vector.keys():
            if i == 'cart' or i == 'pole' :
                for j in feature_vector[i]:
                    state.append(feature_vector[i][j])
                #print(state)
        return np.asarray(state)

    



    def get_best_twostep_action(self, feature_vector):
            '''
                This function computes the best action to take for two step lookahead
                 and returns it as a string
                :return: string action
                '''
            # Create dict of scores
            # Key is first action, scores are rated by second action in order
            # left, right, up, down, nothing
            best_action = {"left": [None for i in range(5)],
                           "right": [None for i in range(5)],
                           "forward": [None for i in range(5)],
                           "backward": [None for i in range(5)],
                           "nothing": [None for i in range(5)]}

            for action in best_action.keys():
                # left, left
                best_action[action][0] = self.two_step_env(feature_vector, [action, 'left'])
                best_action[action][1] = self.two_step_env(feature_vector, [action, 'right'])
                best_action[action][2] = self.two_step_env(feature_vector, [action, 'forward'])
                best_action[action][3] = self.two_step_env(feature_vector, [action, 'backward'])
                best_action[action][4] = self.two_step_env(feature_vector, [action, 'nothing'])

                
            best_score = best_action['left'][0][0]
            expected_state = best_action['left'][0][1]
            # print("Best score: ", best_score)
            action = 'left'
            next_action = 0
            # return the best scoring action
            for i in best_action.keys():
                for j in range(len(best_action[i])):
                    # print(best_action[i][j])
                    if best_action[i][j][0] < best_score:
                        best_score = best_action[i][j][0]
                        action = i
                        next_action = j
                        expected_state = best_action[i][j][1]
            self.lastscore=best_score
            # if(best_score > 1):

            #if we are forcing actions because we are searching for a mapping
            if(self.force_action>=0 and self.force_action < 5):
                action = self.actionnum_to_string(self.force_action)
                expected_state = best_action[action][4][1]
            ecart_x, ecart_y = expected_state["cart"]["x_position"],  expected_state["cart"]["y_position"]            
            


            second_action = ["left", "right", "forward", "backward", "nothing"]
            # rest of storing of history  emebdded in two_step_env wehre its more effiicent
            if(self.force_action >=0 and self.force_action < 5):
                self.action_history[self.force_action][0] = self.format_data(expected_state)                                
            self.reset(feature_vector)# put us back in the state we started.. stepping messed with our state
            cart_x, cart_y = feature_vector["cart"]["x_position"],  feature_vector["cart"]["y_position"]            
            if(self.tbdebuglevel>1): print("Best Two step action ", action, " score ", best_score, " from ", cart_x, cart_y, " by ", ecart_x-cart_x, ecart_y-cart_y)             
            return action, second_action[next_action], expected_state

    def two_step_env(self, feature_vector, steps):
            '''
            Step the environment with the given steps
            :param env:
            :param feature_vector:
            :param steps:
            :return: Score
                                       '''
            if(self.tbdebuglevel>1): print("Two step ", feature_vector) 
            self.reset(feature_vector)
            self.step(steps[0])

            nextstate = self.get_state()
            if(self.string_to_actionnum(steps[0]) == self.force_action):
                print("Force action in two", self.force_action)                
                self.action_history[self.force_action][0] = self.format_data(nextstate)                
            
            if(self.tbdebuglevel>1): print("Try action", steps[0], nextstate)                            
            self.reset(nextstate)
            self.step(steps[1])
            p = self._p            
            p.stepSimulation() #then do nothing for 1 time step so pushes take affect.. no action actually moves cart just changes velocity..
            if(self.tbdebuglevel>1): print("Try action", steps[1], self.get_state())                                        
#            return [self.get_score(self.get_state()), nextstate]
            return [self.get_score(self.get_state()), self.get_state()]        


        #structured like the two-step but that was expensive so teting one steo to see gain vs cost
    def get_best_onestep_action(self, feature_vector):
            '''
                This function computes the best action to take for two step lookahead
                 and returns it as a string
                :return: string action
                '''
            # Create dict of scores
            # Key is first action, scores are rated by second action in order
            # left, right, up, down, nothing
            best_action = {"left": [None for i in range(1)],
                           "right": [None for i in range(1)],
                           "forward": [None for i in range(1)],
                           "backward": [None for i in range(1)],
                           "nothing": [None for i in range(1)]}




            for action in best_action.keys():
                # left, 0
                best_action[action][0] = self.one_step_env(feature_vector, [action, 'nothing'])
                

            best_score = best_action['left'][0][0]
            expected_state = best_action['left'][0][1]
            action = 'left'
            # return the best scoring action
            for i in best_action.keys():
                for j in range(len(best_action[i])):
                    # print(best_action[i][j])
                    if best_action[i][j][0] < best_score:
                        best_score = best_action[i][j][0]
                        action = i
                        expected_state = best_action[i][j][1]

            #if we are forcing actions because we are searching for a mapping
            if(self.force_action>=0 and self.force_action < 5):
                print("Force action in one", self.force_action)
                action = self.actionnum_to_string(self.force_action)
                expected_state = best_action[action][0][1]
                self.action_history[self.force_action][0] = self.format_data(expected_state)                
                        


            self.lastscore=best_score
            self.reset(feature_vector)# put us back in the state we started.. stepping messed with our state        

            return action, "nothing", expected_state

    def one_step_env(self, feature_vector, steps):
            '''
            Step the environment with the given steps
            :param env:
            :param feature_vector:
            :param steps:
            :return: Score
            '''
            if(self.tbdebuglevel>1): print("One step ", feature_vector)             
            self.reset(feature_vector)
            self.step(steps[0])
            state = self.get_state()
            p = self._p            
            p.stepSimulation() #then do nothing for 1 time step so pushes take affect.. no action actually moves cart just changes velocity..
            if(self.tbdebuglevel>1): print("Try action", steps[0],state)                                                    
            return [self.get_score(state), state]





#####  Start domain depenent  adapter (its built into scoring)

    def get_score(self, feature_vector):
            '''
            Score the current state of the environment.
            :return: float score
            '''

            cartpos = [feature_vector["cart"]["x_position"],  feature_vector["cart"]["y_position"],feature_vector["cart"]["z_position"]]
            cartvel = [feature_vector["cart"]["x_velocity"],  feature_vector["cart"]["y_velocity"],feature_vector["cart"]["z_velocity"]]            


            p = self._p
            _, _, _, _, _, ori, _, _ = p.getLinkState(self.cartpole, 1, 1)
            eulers = p.getEulerFromQuaternion(ori)
            pole_x =  abs(eulers[0])
            pole_y =  abs(eulers[1])
            pole_z =  abs(eulers[2])        

            #  we weight the larger of the two errors more, but still consider the other
            maxangle = max(abs(pole_x), abs(pole_y))
            minangle = min(abs(pole_x), abs(pole_y))
            slack = (self.angle_limit-maxangle)/self.angle_limit  #this term is 1 when balance and gets smaller as we get close to failure.  We take 1/slack**4 as a penlty so we big if we get close
            if(slack < .0000001): slackcost = 1e8
            else:
                slackcost = 100/(slack**2)

            cost =0


            collision_penalty  =0  # have to honor slack constraints and don't use react if we have no slack
            mindist=999
            cartspeed=0
            ldist = 999 
            mangle = -999


            #If we don't hav a lot of slack, then ignore collision as we are more likely to die from pole angle
            if(slack < .4):
                # not enough slack.. if doing reactions.. stop it
                self.reactstep = len(self.avoid_list[0])+1
            else:


                #if we have enough slack we consider potential colisions and attack vectors.
                #TB let's use collision engine to get closes distances to account for geometry and keep away from moving blocks

                # start with min distance to walls which are at +- 5
                mindist = 10

                for ablock in self.blocks:
                    nearpoints =  p.getClosestPoints(self.cartpole, ablock,100)
                    for c in nearpoints:           
                        contactdist = c[8]
                        if(contactdist <0):
                            self.char += "CP"                         
    #                        print("Watchout ", self.char)
                        mindist = min(mindist,contactdist)


                #see if  "distance" from  trajectory of  blocks would be an impact.. 
                for block in feature_vector["blocks"]:
                    bpos = [block["x_position"], block["y_position"],block["z_position"]]
                    bvel = [block["x_velocity"], block["y_velocity"],block["z_velocity"]]
                    pdiff = np.subtract(cartpos ,bpos)
                    nval = np.linalg.norm(bvel)
                    if(nval >.01) :
                        dist =  (np.linalg.norm(np.cross(bvel,pdiff))/ nval)
                        if(dist <1.5):
                            if(  self.use_avoid_reaction and (self.reactstep==0 or self.reactstep >= len(self.avoid_list[0]))):
                                self.reactstep=0
                                # get angle so we can decide how to run.. 
                                mangle = 180*math.atan2(pdiff[1],pdiff[0])/3.14159
                                if((mangle >-45 and 45 <= mangle)
                                   or  mangle <-135 or  mangle > 135  ):
                                    if(pole_y > 0):
                                        self.avoid_actions= self.avoid_list[3]
                                    else:
                                        self.avoid_actions= self.avoid_list[2]                                    
                                else:
                                    if(pole_x >0):                                
                                        self.avoid_actions= self.avoid_list[1]
                                    else:
                                        self.avoid_actions= self.avoid_list[0]
 #                               print("Mangle x y ", mangle, pdiff[0], pdiff[1], " Avoid with ",self.avoid_actions);
                                self.char += "HA"                                
                            else:
                                self.char += "SA"
                                cost += 300
#                        print("Block ldist reactstep", dist, self.char, self.reactstep)                                


                    else: dist = ldist
                    ldist = min(dist,ldist)


                #if cart is moving enough we don't worry about as much line-based collision, only contact collision.. if we push to hard pole will fall
                cartspeed = np.linalg.norm(cartvel) 
                if(cartspeed> ldist): ldist = ldist * cartspeed                 
                mindist = min(ldist,mindist)
            
            
            
                if(mindist > 1 and mindist < 3):
                    collision_penalty = (3-mindist)         #a  penalty minimal menalty if we step in direction that is close to collisoin region
                elif(mindist > 0):
                    collision_penalty = 2+  1/( mindist*mindist)         #a  penalty that get's get very large as we get close , but enough power far away to avoid if we can


            

            cost  += slackcost    + ( maxangle)**2 + (minangle)**2 +   collision_penalty                


            if(False and cost > 1000):
#            if(self.tbdebuglevel>1): 
                print(" cost,  slack, slackcost  ", round(cost,8), round(slack,3),round(slackcost,3), "pole xyz", round(abs(pole_x),3), round(abs(pole_y),3),round(abs(pole_z),3),
                      "  dists ", round(collision_penalty,3),round(mindist,3),   "angle limit", round(self.angle_limit,3), "  At", self.tick,"score=", round(cost,2),"at",round(cartpos[0],2), round(cartpos[1],2))

            return cost


            # get best action.. if our prediciton probablities (arg)  are good, use one step, else use two
        #this is where we should try to adapt physics parmeters if things are going badly.. 
    def get_best_action(self, feature_vector, prob=0):
        if(self.use_avoid_reaction and self.reactstep >= 0  and self.reactstep < len(self.avoid_actions)):
            react = self.avoid_actions[self.reactstep]
            self.char += "AV"+str(self.reactstep)                                            
            print("Avoiding @reactstep", self.reactstep, " with ", react)
            score,nstate = self.one_step_env(feature_vector, [react , 'nothing'])
            state = [react,"nothing",nstate]
            self.reactstep += 1
            self.lastscore = -1.0

        else:

    #       if we have colission potential for any action (char != "") so do two-step action search
    #       if we have low score  we can go faser uding one-setp
           if(  (self.lastscore > 0 and  ((prob < .49  and self.lastscore < 300) or (self.lastscore < 200) ))):            # make it mroe often just one making it faster
                state= self.get_best_onestep_action(feature_vector)
                if(self.tbdebuglevel>-1): print("Best one score", self.lastscore)
           else:
                state= self.get_best_twostep_action(feature_vector)
                if(self.tbdebuglevel>-1): print("Best two score", self.lastscore)
           #we do tick here to update one timestep..
        self.tick = self.tick + 1
        if(self.tbdebuglevel>2):
            print("Expected state", state)           
        return state
#####  end domain depenent  adapter (its built into scoring)
