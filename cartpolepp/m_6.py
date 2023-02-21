import math
import numpy as np
import os.path

from .cartpoleplusplus import CartPoleBulletEnv


class CartPolePPMock6(CartPoleBulletEnv):

    def __init__(self, difficulty, params: dict = None):
        super().__init__(params=params)

        self.difficulty = difficulty

    def block_repulsion_forces(self, positions):
        # Calculate the net force on each block due to repulsion from other blocks
        forces = []
        for i, pos_i in enumerate(positions):
            force = [0, 0, 0]
            for j, pos_j in enumerate(positions):
                if i == j:
                    continue
                dist = np.linalg.norm(np.asarray(pos_i) - np.asarray(pos_j))
                if dist > 0:
                    direction = (np.asarray(pos_j) - np.asarray(pos_i)) / dist
                    force -= direction * (self.block_attraction / dist)
            forces.append(force)
        return forces

    def step(self, action):
        # Run one step of simulation
        p = self._p

        # Calculate forces on blocks due to repulsion from other blocks
        block_positions = [p.getBasePositionAndOrientation(b)[0] for b in self.blocks]
        block_forces = self.block_repulsion_forces(block_positions)

        # Calculate forces on blocks due to gravity
        grav_forces = [self.block_mass * np.array((0, 0, -self._g))] * len(self.blocks)

        # Calculate total forces on blocks
        total_forces = [bf + gf for bf, gf in zip(block_forces, grav_forces)]

        # Apply forces to blocks
        for b, f in zip(self.blocks, total_forces):
            p.applyExternalForce(b, -1, f, (0, 0, 0), p.LINK_FRAME)

        # Run simulation step
        p.stepSimulation()

        # Get state
        state = self.get_state()

        # Check if done
        done = self.done(state)

        # Get reward
        reward = self.reward(state, action)

        return state, reward, done, {}
