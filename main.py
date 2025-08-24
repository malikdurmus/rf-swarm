from args import Args
import visualization
import backend
from simulation import Simulation
import tyro
from args import Args


def main():
    """
    Main function to set up and run the drone simulation.
    """
    # 1. Setup
    args = tyro.cli(Args)
    
    # --- THIS IS THE KEY NEW LINE ---
    # Set the backend (CPU or GPU) for the entire application
    backend.set_backend(args.gpu)
    # --------------------------------

    sim = Simulation(size=args.size, emitter_pos=args.emitter_pos)
    sim.generate_drones(args.num_drones)
    
    # Use the 'to_numpy' helper for printing, as the array might be on the GPU
    print(f"Emitter is at: {visualization.to_numpy(sim.emitter_pos)}")
    for i, drone in enumerate(sim.drones):
        print(f"Agent{i} starts at: {visualization.to_numpy(drone.pos)}")

    # 2. Initial Visualization
    visualization.setup_plotting()
    visualization.plot_simulation_state(sim, title="Initial Belief (Uniform)")
    input("Press Enter to start the simulation...")

    # 3. Main Loop
    for t in range(1000):
        sim.run_sim_step()
        
        # Periodically update the plot
        if t % 10 == 0:
            visualization.plot_simulation_state(sim, title=f"Belief After Measurement (t={t})")

if __name__ == "__main__":
    main()
