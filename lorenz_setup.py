from scipy.integrate import odeint
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from lorenz_settings import N_UNITS, MIN_DELAY, MAX_DELAY, DELAY_DIST_OFFSET


def sample_delays(
    n_in,
    n_out,
    min_delay=MIN_DELAY,
    max_delay=MAX_DELAY,
    offset=DELAY_DIST_OFFSET,
    lmbda=1.15,
    return_pmf=False,
):
    axonal_delay_range = np.arange(min_delay, max_delay + 1)
    axonal_delay_pmf = [
        lmbda ** (k / 8) * np.exp(-lmbda) / math.gamma((k / 8))
        for k in range(offset, max_delay - min_delay + offset + 1)
    ]
    axonal_delay_pmf /= np.sum(axonal_delay_pmf)
    samples = np.random.choice(
        axonal_delay_range, size=n_in * n_out, p=axonal_delay_pmf
    ).reshape(n_out, n_in)
    if return_pmf:
        return (axonal_delay_range, axonal_delay_pmf), samples
    else:
        return samples


def get_lorenz_stimulation(
    sigma=10,
    rho=28,
    beta=8 / 3,
    duration_placement=2000,
    duration_sample=200,
    dt=0.01,
    initial_state=[0, 5, 0],
    n_units=N_UNITS,
    kernel_std=2.0,
):
    # Time points
    t = np.arange(0, duration_placement, dt)

    # Simulate the Lorenz system
    states = simulate_lorenz(t, initial_state, sigma, rho, beta)

    # Perform k-means clustering
    unit_locations = perform_kmeans(states, n_units)

    t_sample = np.arange(0, duration_sample, dt)
    sample_states = simulate_lorenz(t_sample, states[-1, :], sigma, rho, beta)

    # Simulate the units
    unit_activation = simulate_units(
        sample_states, unit_locations, kernel_std=kernel_std
    )

    return states, unit_locations, sample_states, unit_activation


def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def simulate_lorenz(t, initial_state, sigma, rho, beta):
    states = odeint(lorenz, initial_state, t, args=(sigma, rho, beta))
    return states


def perform_kmeans(states, n_units=N_UNITS):
    kmeans = KMeans(n_clusters=n_units, n_init="auto").fit(states)
    unit_locations = kmeans.cluster_centers_
    return unit_locations


def plot_trajectory_and_units(states, unit_locations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.5)
    ax.scatter(
        unit_locations[:, 0], unit_locations[:, 1], unit_locations[:, 2], color="r"
    )
    ax.set_title("Lorenz System Trajectory and Unit Positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def compute_distances(states, unit_locations):
    distances = cdist(states, unit_locations)
    return distances


def compute_probabilities(distances, decay_rate=3.0, scale=100.0):
    probabilities = np.exp(-decay_rate * distances) * scale
    return np.minimum(probabilities, 1)


def simulate_units(states, unit_locations, kernel_std=2.0):
    # Compute the distances from each state to each unit
    distances = compute_distances(states, unit_locations)
    """
    if spiking:
        # Compute the probabilities
        probabilities = compute_probabilities(distances)
        active_units = np.zeros((probabilities.shape[0], n_units))

        random_numbers = np.random.uniform(size=(probabilities.shape[0], n_units))
        active_units = (random_numbers <= probabilities).astype(int)
        return active_units
    """
    activity = norm.pdf(distances, loc=0, scale=kernel_std)
    activity /= activity.max()
    return activity


def animate_activations(unit_locations, states, unit_activation):
    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    # Plot the initial trajectory and unit locations
    # trajectory, = ax.plot([], [], [], lw=0.5)
    units = ax.scatter(
        unit_locations[:, 0], unit_locations[:, 1], unit_locations[:, 2], c=[]
    )

    # Set the title and labels
    ax.set_title("Lorenz Unit Activations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Function to update the plot for each frame
    def update(i):
        # Update the trajectory
        # trajectory.set_data(states[:i+1, 0], states[:i+1, 1])
        # trajectory.set_3d_properties(states[:i+1, 2])

        # Update the unit colors based on their activation state
        units.set_array(unit_activation[i, :])

        return units
        # return trajectory, units

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(states), interval=5)
    return fig, anim
