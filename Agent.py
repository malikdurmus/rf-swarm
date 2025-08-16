import numpy as np

class Agent:
    def __init__(self, sim_size):
        self.sim_size = sim_size
        # random position (x,y) for the agent (drone)
        self.agent_pos = np.random.randint(0, sim_size, size=2, dtype=int)

        # create the initial belief map
        # (each position has an initial value of 1 / total cells)
        self.belief_grid = (np.ones(self.sim_size, self.sim_size)
                            / self.sim_size ** 2)

