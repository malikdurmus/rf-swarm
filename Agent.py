import numpy as np
import cupy as cp

moves = {
            "North": [0, 1],
            "South": [0, -1],
            "East": [1, 0],
            "West": [-1, 0]
        }


class Agent:
    def __init__(self, sim_size):
        self.sim_size = sim_size
        # random position (x,y) for the agent (drone)
        self.pos = np.random.randint(0, sim_size, size=2, dtype=int)

    def _valid_pos(self, pos):
        print(        self.sim_size - 1 > pos[0] > 0 and  self.sim_size - 1 > pos[1]  > 0)
        return self.sim_size - 1 > pos[0] > 0 and  self.sim_size - 1 > pos[1]  > 0

    def random_move(self):
        direction = np.random.randint(0, 4)
        # --- Y-axis movement (North/South) ---
        if direction == 0 and self.pos[1] < self.sim_size - 1:  # North
            self.pos[1] += 1
        elif direction == 2 and self.pos[1] > 0:  # South
            self.pos[1] -= 1
        # --- X-axis movement (East/West) ---
        elif direction == 1 and self.pos[0] < self.sim_size - 1:  # East
            self.pos[0] += 1
        elif direction == 3 and self.pos[0] > 0:  # West
            self.pos[0] -= 1

    def greedy_move(self,sim):
        """ The agent will move to the next position that gives it the strongest signal"""

        best_pos = self.pos
        max_rssi = sim.get_rssi(self.pos)

        for k, v in moves.items():
            next_pos = self.pos + v
            if self._valid_pos(next_pos):
                next_rssi = sim.get_rssi(next_pos)
                if next_rssi > max_rssi:
                    max_rssi = next_rssi
                    best_pos = next_pos

        self.pos = best_pos
     # alternative: move to the position where you have the highest prob (likelihood): TODO: (does this make sense?)






def update_belief_gpu(belief_grid, rssi_measurement, drone_pos):

    belief_grid_gpu = cp.asarray(belief_grid)
    drone_pos_gpu = cp.asarray(drone_pos)

    # two 1000x1000 matrices
    height, width = belief_grid_gpu.shape
    x_indices = cp.arange(width)
    y_indices = cp.arange(height)
    x_coords, y_coords = cp.meshgrid(x_indices, y_indices)

    # Shape (1000, 1000, 2)
    possible_emitter_pos = cp.stack([y_coords, x_coords], axis=-1)

    # The distance from the drone to the coordinate
    dist = cp.linalg.norm(drone_pos_gpu - possible_emitter_pos, axis=-1)
    dist = cp.maximum(dist, 0.001)  # avoid log(0)

    # Expected RSSI (The drone would get this rssi, if this point was really the emitter)
    expected_rssi = -20 * cp.log10(dist) - 30

    # The likelihood -->   P( drone seeing the RSSI | Radar is at this point)
    # we assume normal distribution TODO: is this assumption justified?
    likelihood = _gpu_norm_pdf(rssi_measurement, loc=expected_rssi, scale=2)

    # Calculate the posterior -> P (Radar | RSSI)
    belief_grid_gpu *= likelihood

    # Normalize the grid to make the probs sum up to 1
    belief_grid_gpu /= cp.sum(belief_grid_gpu)

    return belief_grid_gpu.get()

def _gpu_norm_pdf(x, loc, scale):
    """A CUDA-powered Gaussian PDF calculator."""
    pi = 3.1415926535
    return (1.0 / (scale * cp.sqrt(2 * pi))) * cp.exp(-0.5 * ((x - loc) / scale)**2)



# @staticmethod
# def update_belief(belief_grid, rssi_measurement, drone_pos):
#     for (x,y) , value in np.ndenumerate(belief_grid):
#         possible_emitter_pos = np.array([x, y])
#         # Calculate the distance from our drone to this hypothetical emitter position
#         dist = np.linalg.norm(drone_pos - possible_emitter_pos)
#         if dist == 0:
#             dist = 0.1 # Avoid log(0) issues
#
#         # What signal strength would we EXPECT to get if the emitter was here?
#         expected_rssi = -20 * np.log10(dist) - 30
#         # we assume normal distribution
#         likelihood = norm.pdf(rssi_measurement, loc=expected_rssi, scale=2)
#
#         belief_grid[x, y] *= likelihood
#
#     belief_grid = belief_grid / np.sum(belief_grid)

