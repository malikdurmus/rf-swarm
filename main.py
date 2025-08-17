import time

from matplotlib import pyplot as plt
from Simulation import Simulation

@staticmethod
def plot_belief_grid():
    # Plot the belief grid heatmap
    plt.imshow(sim.belief_grid, origin='lower')
    plt.title("Belief After Measurement")
    # Plot the emitter's true position (yellow star)

    # Use column for x, row for y
    plt.plot(sim.emitter_pos[0], sim.emitter_pos[1], 'y*', markersize=15, label='Emitter (True)')

    # Consolidate labels in the legend by handling them outside the loop
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    # Plot the drones' positions (red 'X')
    for drone in sim.drones:
        # Use column for x, row for y
        plt.plot(drone.pos[0], drone.pos[1], 'rx', markersize=10, label='Drone')


sim = Simulation(size=1000)

sim.generate_drones(4)


print(f"Emitter is at: {sim.emitter_pos}")
for i in range(len(sim.drones)):
    print(f"Agent{i} starts at: {sim.drones[i].pos}")


# Let's plot the initial, uniform belief
plt.imshow(sim.belief_grid, origin='lower')
plt.title("Initial Belief (Uniform)")
plt.show()


for t in range (500):

    sim.run_sim()
    if t % 5 == 0:
        time.sleep(0.1)
        plot_belief_grid()