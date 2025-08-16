import numpy as np
from scipy.stats import norm

class Simulation:

    def __init__(self, size=100):
        self.size = size

        # drones/agents
        self.drones = []

    def _place_emitter(self):
        # random position (x,y) for the emitter (radar)
        self.emitter_pos = np.random.randint(0, self.size, size=2, dtype=int)


    def get_rssi(self, drone_position):
        pass