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
    U_FACTOR,
    W_FACTOR,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Use GPU if available
ZERO = torch.tensor([0], device=device)

def nonlinearity(x, scale=1e2, offset=0):
    return 1 / (1 + torch.exp(-scale * (x - offset)))

def lateral_inhibition(mean_activity, target_activity):
    return torch.maximum(mean_activity - target_activity, ZERO)


def simulate(
    input_current,
    w,
    axonal_delays,
    inh_delays,
    u_factor=U_FACTOR,
    stimulate=1,
    w_factor=W_FACTOR,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
    fast_gamma=1,
    X_init=None,
    lmbda1=LMBDA1,
    lmbda2=LMBDA2,
    lmbda3=LMBDA3,
    f=nonlinearity,
    dt=DT,
    n_units=N_UNITS,
    init_steps=MAX_DELAY,
    w_inh=None,
    inh_factor=1,
    target_activity=0.01
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Use GPU if available

    input_current = torch.from_numpy(input_current).to(device)
    if X_init is not None:
        X_init = torch.from_numpy(X_init).to(device)
    
    w = torch.from_numpy(w).to(device)
    if w_inh is not None:
        w_inh = torch.from_numpy(w_inh).to(device)
    else:
        w_inh = torch.zeros_like(w).to(device)
    
    n_steps = input_current.shape[0]
    X = torch.zeros((n_steps, n_units), device=device)
    adaptation = torch.ones(n_units, device=device) * target_activity
    fast_adaptation = torch.ones(n_units, device=device) * target_activity

    A = []
    fast_A = []

    def get_dx(t, init_stim, I_in=None):
        return (
            (
                -lmbda1 * X[t - 1]
                + f(
                    (init_stim | stimulate) * u_factor * input_current[t - 1]
                    + (w_factor * I_in if not init_stim else 0)
                    - lmbda3 * lateral_inhibition(fast_adaptation.mean(), target_activity)
                ) / ((adaptation + adaptation.mean()) / (2 * target_activity))**3
            )
            * dt
            * alpha
        )
    
    if X_init is not None:
        X[: X_init.shape[0]] = X_init
    else:
        for t in range(1, init_steps):
            adaptation += (X[t - 1] - adaptation) * dt * gamma
            fast_adaptation += (X[t - 1] - fast_adaptation) * dt * fast_gamma
            A.append(adaptation.cpu().numpy())
            fast_A.append(fast_adaptation.cpu().numpy())

            S_in_fast = X[t - inh_delays, torch.arange(n_units, device=device)]
            I_in = -inh_factor * (S_in_fast * w_inh).sum(1)
            dx = get_dx(t, True)
            X[t] = X[t - 1] + dx

    W_mean_hist = []
    fast_adaptation += (1 * target_activity - fast_adaptation.mean())
    #adaptation += (1 * target_activity - adaptation.mean())
    
    for t in range(init_steps, n_steps):
        adaptation += (X[t - 1] - adaptation) * dt * gamma
        fast_adaptation += (X[t - 1] - fast_adaptation) * dt * fast_gamma
        A.append(adaptation.cpu().numpy())
        fast_A.append(fast_adaptation.cpu().numpy())

        S_in = X[t - axonal_delays, torch.arange(n_units, device=device)]
        S_in_fast = X[t - inh_delays, torch.arange(n_units, device=device)]
        I_in = (S_in * w).sum(1) - inh_factor * (S_in_fast * w_inh).sum(1)
        dx = get_dx(t, False, I_in)

        dw = (
                -lmbda2 * w
                + S_in * (dx / dt)[:, np.newaxis]
             ) * dt * beta

        X[t] = X[t - 1] + dx
        w += dw
        W_mean_hist.append(w.mean().cpu().numpy())

    return X.cpu().numpy(), w.cpu().numpy(), np.array(W_mean_hist), np.array(A), np.array(fast_A)
