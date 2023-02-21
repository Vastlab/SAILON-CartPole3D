import math
import numpy as np
import os.path

from .cartpoleplusplus import CartPoleBulletEnv


class CartPolePPMock7(CartPoleBulletEnv):

    def __init__(self, difficulty, params: dict = None):
        super().__init__(params=params)

        self.difficulty = difficulty

        # Generate a random point for the pole to be attracted to
        self.target_pos = self.np_random.uniform(low=-4.0, high=4.0, size=(3,))

    def step(self, action):
        p = self._p

        # Convert from string to int
        if action == 'nothing':
            action = 0
        elif action == 'right':
            action = 1
        elif action == 'left':
            action = 2
        elif action == 'forward':
            action = 3
        elif action == 'backward':
            action = 4

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

        # Apply corrected forces
        p.applyExternalForce(self.cartpole, 0, (fx, fy, 0.0), (0, 0, 0), p.LINK_FRAME)

        # Attract pole to the target point
        pole_pos, _ = self._p.getBasePositionAndOrientation(self.pole)
        force = (self.target_pos - np.array(pole_pos)) * 10.0
        self._p.applyExternalForce(self.pole, -1, force, (0, 0, 0), self._p.LINK_FRAME)

        # Run simulation step
        p.stepSimulation()

        # Get state
        state = self.get_state()

        # Check if done
        done = self.done(state)

        # Get reward
        reward = self.reward(state, action)

        return state, reward, done, {}
