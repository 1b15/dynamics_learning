from lorenz_setup import simulate_lorenz

import matplotlib.pyplot as plt

import torch
import numpy as np

from lorenz_settings import (
    N_UNITS,
    MAX_DELAY,
    ALPHA,
    BETA,
    GAMMA,
    DT,
    LMBDA1,
    LMBDA2,
    LMBDA3,
    LMBDA4,
    U_FACTOR,
    W_FACTOR,
)

def preprocess_X(X):
    # normalize features to mean 0 and variance 1
    # return (X - X.mean(axis=0)) / np.std(X, axis=0)
    # normalize features to range [0, 1]
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def generate_tuning_curves(scalars, dimensions):
    """
    Generate tuning curves as Gaussian curves centered around scalar values.

    Args:
    scalars (ndarray): Array of scalar values between 0 and 1.
    dimensions (int): Number of dimensions in the array.
    width (float): Width of the Gaussian curve.

    Returns:
    tuning_curves (ndarray): Array of N-dimensional arrays representing tuning curves.
    """
    width = 1 / dimensions
    positions = np.linspace(-3 * width, 1 + 3 * width, dimensions)
    tuning_curves = np.exp(
        -((positions[np.newaxis, :] - scalars[:, np.newaxis]) ** 2) / (2 * width**2)
    )
    return tuning_curves


def estimate_scalars_from_tuning_curves(tuning_curves):
    """
    Estimate scalar values from arrays of N-dimensional tuning curves.
    
    Args:
    tuning_curves (ndarray): Array of N-dimensional arrays representing tuning curves.
    width (float): Width of the Gaussian curve used to generate the tuning curves.
    
    Returns:
    estimated_scalars (ndarray): Array of estimated scalar values.
    """
    dimensions = tuning_curves.shape[1]
    width = 1/dimensions
    positions = np.linspace(-3 * width, 1 + 3 * width, dimensions)
    
    # Compute the estimated scalar values for each tuning curve
    tuning_curves = tuning_curves.copy()
    tuning_curves += 1e-6
    tuning_curves = tuning_curves / (tuning_curves.sum(1)[:, np.newaxis])
    estimated_scalars = np.sum(positions * tuning_curves, axis=1) / (np.sum(tuning_curves, axis=1))
    return estimated_scalars

def get_X(
    sigma=10,
    rho=28,
    beta=8 / 3,
    duration=2000,
    dt=0.01,
    initial_state=[0, 5, 0],
    n_units_p=30,
):
    # Time points
    t = np.arange(0, duration, dt)

    # Simulate the Lorenz system
    X_raw = preprocess_X(simulate_lorenz(t, initial_state, sigma, rho, beta))

    X = np.array([generate_tuning_curves(X_raw[:, i], n_units_p) for i in range(3)])
    return X_raw, np.transpose(X, (1, 0, 2))

X_raw, X = get_X(duration=50000)

def nonlinearity(x, scale=10, offset=0.5):
    return 1 / (1 + torch.exp(-scale * (x - offset)))

def f_homeostasis(mean_activity, threshold_offset=0.0):
    return max(mean_activity + threshold_offset, 0)

from matplotlib.animation import FuncAnimation


def animate_scatter_and_line(positions, trajectory):
    T, _, N = positions.shape
    M, _ = trajectory.shape

    # Create a figure and 3D axis for the scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter([], [], [], c="b", marker="o")
    (line,) = ax.plot(
        [], [], [], c="r", linewidth=0.5, alpha=0.5
    )  # Static line trajectory

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Update function for animation
    def update(frame):
        x = positions[frame, 0, :]
        y = positions[frame, 1, :]
        z = positions[frame, 2, :]
        scatter._offsets3d = (x, y, z)

        line.set_data(trajectory[:, 0], trajectory[:, 1])
        line.set_3d_properties(trajectory[:, 2])

        return scatter, line

    # Create the animation
    ani = FuncAnimation(fig, update, frames=T, blit=True, interval=5)
    return fig, ani


def simulate(
    X,
    lmbda1=1,
    lmbda2=6e1,
    n_units=N_UNITS,
    dt=DT,
    alpha=5e2,
    beta_in=5e-4,
    gamma1=1,
    gamma2=2e2,
    target_activity=20/N_UNITS,
    w=None,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Use GPU if available

    n_steps, n_features, n_units_p = X.shape

    # w = 1 * np.random.randn(n_units, n_features, n_units_p)
    # w -= w.mean()
    if w is None:
        #w = np.random.uniform(0, 0.5, size=(n_units, n_features, n_units_p))**2  # + 0.1
        w = generate_tuning_curves(np.random.rand(n_units * n_features), n_units_p).reshape(n_units,n_features,-1) / 8

    V = torch.zeros((n_steps, n_units), device=device)
    X = torch.from_numpy(X).to(device)
    w = torch.from_numpy(w).to(device)
    slow_a = target_activity * torch.ones(n_units, device=device)
    fast_a = target_activity * torch.ones(n_units, device=device)
    #lateral_inhibition_matrix = (torch.ones(n_units, n_units) - torch.eye(n_units)).to(device) / n_units
    zeros = torch.zeros(n_units, device=device)

    W = []
    W_end = []
    W_mean_hist = []
    W_var_hist = []
    DW_max = []
    Slow_A = []
    Fast_A = []

    for t in range(-5000, 0):
        dv = (
            (
                - V[t - 1]
                + lmbda1 * nonlinearity(
                    torch.einsum("kij,ij->k", w, X[t])
                    + lmbda2 * min(target_activity - fast_a.mean(), 0)
                )
            )
            * dt
            * alpha
        )

        V[t] = V[t - 1] + dv
        slow_a += (V[t] - slow_a) * dt * gamma1
        fast_a += (V[t] - fast_a) * dt * gamma2

    for t in range(n_steps):
        dv = (
            (
                - V[t - 1]
                + lmbda1 * nonlinearity(
                    torch.einsum("kij,ij->k", w, X[t])
                    + lmbda2 * min(target_activity - fast_a.mean(), 0)
                )
            )
            * dt
            * alpha
        )

        V[t] = V[t - 1] + dv

        dw = (
            (
                -2.5e1 * (
                    #(w / w.mean((1, 2))[:, None, None]) *
                    torch.maximum(
                        slow_a - target_activity, zeros
                    )[:, np.newaxis, np.newaxis]
                )
                + torch.einsum("kij,k->kij", (X[None, t] - w), dv / dt + V[t - 1] / dt / alpha)
                - 1.2e2 * w**3
                + 10 * torch.einsum(
                    "kij,k->kij",
                    (X[None, t] - w),
                    torch.maximum(
                        (target_activity - fast_a), zeros
                    )
                    * torch.maximum(
                        (target_activity - slow_a), zeros
                    )
                    # * torch.rand(n_units, device=device)
                )
            )
            * dt
            * beta_in
        )

        w += dw
        w = torch.relu(w)
        slow_a += (V[t] - slow_a) * dt * gamma1
        fast_a += (V[t] - fast_a) * dt * gamma2

        if n_steps - t < 5000:
            W_end.append(w.cpu().numpy())
        if t % 5000 == 0:
            W.append(w.cpu().numpy())

        DW_max.append(dw.abs().max().cpu().item())
        Slow_A.append(slow_a.mean().cpu().item())
        Fast_A.append(fast_a.mean().cpu().item())

    return (
        V.cpu().numpy(),
        w.cpu().numpy(),
        np.array(W),
        slow_a.cpu().numpy(),
        DW_max,
        Slow_A,
        Fast_A,
        W_end
    )
V, w_new, W, a, DW_max, Slow_A, Fast_A, W_end = simulate(X[:], n_units=600)#, w=w_new.copy())

import os

if not os.path.exists("checkpoints/kmeans"):
    os.makedirs("checkpoints/kmeans")

np.save("checkpoints/kmeans/X.npy", X)
np.save("checkpoints/kmeans/X_raw.npy", X_raw)
np.save("checkpoints/kmeans/w.npy", w_new)
np.save("checkpoints/kmeans/W.npy", W)
np.save("checkpoints/kmeans/W_end.npy", W_end)
np.save("checkpoints/kmeans/V.npy", V)
