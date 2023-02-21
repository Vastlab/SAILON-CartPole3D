import math
import numpy as np
import os.path

from .cartpoleplusplus import CartPoleBulletEnv


class CartPolePPMock8(CartPoleBulletEnv):
    def __init__(self, difficulty, params: dict = None):
        super().__init__(params=params)
        self.difficulty = difficulty
        self.block_attraction = 20.0  # strength of block attraction to pole

    def step(self, action):
        # Run one step of simulation
        p = self._p

        # Calculate forces on blocks due to attraction to other blocks
        block_positions = [p.getBasePositionAndOrientation(b)[0] for b in self.blocks]
        block_forces = self.block_attraction_forces(block_positions)

        # Calculate forces on blocks due to gravity
        grav_forces = [self.block_mass * np.array((0, 0, -self._g))] * len(self.blocks)

        # Calculate forces on blocks due to attraction to pole
        pole_pos, _ = p.getBasePositionAndOrientation(self.cartpole)
        pole_forces = []
        for block in self.blocks:
            block_pos, _ = p.getBasePositionAndOrientation(block)
            direction = np.array(pole_pos) - np.array(block_pos)
            distance = np.linalg.norm(direction)
            if distance > 0:
                force = direction / distance * self.block_attraction
                pole_forces.append(force)
            else:
                pole_forces.append(np.zeros(3))

        # Calculate total forces on blocks
        total_forces = [bf + gf + pf for bf, gf, pf in zip(block_forces, grav_forces, pole_forces)]

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
