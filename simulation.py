import backend as bk
import numpy as np


moves = {
    "North": np.array([0, 1]),
    "South": np.array([0, -1]),
    "East": np.array([1, 0]),
    "West": np.array([-1, 0])
}

class Agent:
    """Represents a single drone in the simulation."""
    def __init__(self, sim_size):
        self.sim_size = sim_size

        self.pos = bk.xp.random.randint(0, sim_size, size=2, dtype=int)

    def _valid_pos(self, pos):
        """Checks if a position is within the simulation bounds."""

        return 0 < pos[0] < self.sim_size - 1 and 0 < pos[1] < self.sim_size - 1

    def greedy_move(self, sim):
        """Moves the agent to the adjacent cell with the strongest signal."""
        best_pos = self.pos
        max_rssi = sim.get_rssi(self.pos)

        for _, move_vector in moves.items():
            next_pos = self.pos + move_vector
            if self._valid_pos(next_pos):
                next_rssi = sim.get_rssi(next_pos)
                if next_rssi > max_rssi:
                    max_rssi = next_rssi
                    best_pos = next_pos
        self.pos = best_pos

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

class Simulation:
    """Manages the state of the simulation, including the grid and agents."""

    def __init__(self, size, emitter_pos):
        self.size = size
        self.drones = []
        
        # Use the backend 'xp' for all array initializations
        self.belief_grid = bk.xp.ones((size, size), dtype=bk.xp.float32) / (size ** 2)
        self._place_emitter(emitter_pos)

    def _place_emitter(self, emitter_pos):
        if emitter_pos == None: 
            self.emitter_pos = np.random.randint(0,self.size,size =2,dtype=int)
        else: 
            self.emitter_pos = bk.xp.array(emitter_pos, dtype=int)



    def generate_drones(self, number):
        """Creates and adds a specified number of drones to the simulation."""
        self.drones.extend([Agent(sim_size=self.size) for _ in range(number)])


    def get_rssi(self, drone_position):

        # Euclidean distance l2
        dist = bk.xp.linalg.norm(drone_position - self.emitter_pos)

        # Prevent division by zero if a drone is exactly on the emitter
        if dist == 0:
            return -10.0  # A very strong signal

        # Calculate signal strength using inverse square law
        # The '20' is a path loss exponent, '-30' is a reference signal strength.
        # TODO: Use/Implement TDOA given Avaliable GPS
        signal_strength = -20 * bk.xp.log10(dist) - 30

        # Add random noise
        # Gaussian noise with a standard deviation of 2 dBm
        noise = bk.xp.random.normal(0, 2)

        return signal_strength + noise


    def _norm_pdf(self, x, loc, scale):
        """A backend-agnostic Gaussian PDF calculator."""
        pi = 3.1415926535
        return (1.0 / (scale * bk.xp.sqrt(2 * pi))) * bk.xp.exp(-0.5 * ((x - loc) / scale)**2)

    def update_belief(self, rssi_measurement, drone_pos):
        """
        Vectorized belief grid update using the selected backend (CPU or GPU).
        """
        height, width = self.belief_grid.shape
        x_indices = bk.xp.arange(width)
        y_indices = bk.xp.arange(height)
        x_coords, y_coords = bk.xp.meshgrid(x_indices, y_indices)

        # Create a grid of all possible emitter positions
        possible_emitter_pos = bk.xp.stack([y_coords, x_coords], axis=-1)

        # Calculate distance from the drone to every cell in the grid at once
        dist = bk.xp.linalg.norm(drone_pos - possible_emitter_pos, axis=-1)
        dist = bk.xp.maximum(dist, 0.001)  # Avoid log(0)

        # Calculate expected RSSI for every cell
        expected_rssi = -20 * bk.xp.log10(dist) - 30

        # Calculate likelihood P(measurement | emitter at cell) for all cells
        likelihood = self._norm_pdf(rssi_measurement, loc=expected_rssi, scale=2)

        # Update belief grid (Bayes' rule)
        self.belief_grid *= likelihood
        self.belief_grid /= bk.xp.sum(self.belief_grid) # Normalize

    def run_sim_step(self):
        """Runs one iteration of the simulation for all drones."""
        for drone in self.drones:
            rssi = self.get_rssi(drone.pos)
            self.update_belief(rssi_measurement=rssi, drone_pos=drone.pos)
            drone.greedy_move(self)
