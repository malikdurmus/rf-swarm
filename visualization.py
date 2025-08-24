import matplotlib.pyplot as plt
from simulation import Simulation  # Import for type hinting
import backend as bk # Import our backend to check which one is active

def to_numpy(arr):
    """
    Converts a CuPy array to a NumPy array.
    If the array is already a NumPy array, it returns it unchanged.
    """
    # Check if the GPU backend is active and if the array is a CuPy array
    if bk.IS_GPU_BACKEND and isinstance(arr, bk.xp.ndarray):
        return arr.get()
    return arr

def setup_plotting():
    """Activates interactive plotting mode."""
    plt.ion()

def plot_simulation_state(sim: Simulation, title: str):
    """Clears the current plot and draws the new state of the simulation."""
    plt.clf()

    # Convert data to numpy arrays
    belief_grid_np = to_numpy(sim.belief_grid)
    emitter_pos_np = to_numpy(sim.emitter_pos)

    # Plot the belief grid heatmap
    plt.imshow(belief_grid_np, origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    
    # Plot the emitter's true position
    plt.plot(emitter_pos_np[0], emitter_pos_np[1], 'y*', markersize=15, label='Emitter (True)')

    for i, drone in enumerate(sim.drones):
        drone_pos_np = to_numpy(drone.pos)
        label = 'Drone' if i == 0 else "" # Label only the first drone
        plt.plot(drone_pos_np[0], drone_pos_np[1], 'rx', markersize=10, label=label)
            
    plt.legend()
    plt.pause(0.001)
