import numpy as np
from Agent import Agent , update_belief_gpu


class Simulation:

    def __init__(self, size=100):
        self.size = size

        # create the initial belief map
        # (each position has an initial value of 1 / total cells)
        self.belief_grid = (np.ones((size, size))
                            / size ** 2)

        # drones/agents
        self.drones = []

        self._place_emitter()



    def add_drone(self, drone):
        self.drones.add(drone)

    def _get_emitter_pos(self):
        return self.emitter_pos

    def _place_emitter(self):
        # random position (x,y) for the emitter (radar)
        self.emitter_pos = np.random.randint(0, self.size, size=2, dtype=int)

    def get_rssi(self, drone_position):

        # Euclidean distance l2
        dist = np.linalg.norm(drone_position - self.emitter_pos)

        # Prevent division by zero if a drone is exactly on the emitter
        if dist == 0:
            return -10.0  # A very strong signal

        # Calculate signal strength using inverse square law
        # The '20' is a path loss exponent, '-30' is a reference signal strength.
        # TODO: Use/Implement TDOA given Avaliable GPS
        signal_strength = -20 * np.log10(dist) - 30

        # Add random noise
        # Gaussian noise with a standard deviation of 2 dBm
        noise = np.random.normal(0, 2)

        return signal_strength + noise

    def generate_drones(self,number):
        # Generate n drones
         self.drones.extend([Agent(sim_size = self.size) for _ in range(number)])

    def run_sim(self):

        for drone in self.drones:
            rssi = self.get_rssi(drone.pos)
            print(f"Agent measured RSSI: {rssi:.2f} dBm")
            self.belief_grid = update_belief_gpu(rssi_measurement = rssi,
                          drone_pos=drone.pos,
                          belief_grid=self.belief_grid)
            # drone.random_move()
            drone.greedy_move(self)