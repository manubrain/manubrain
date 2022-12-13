""" A synthetic data-set for debugging.
Based on: https://github.com/v0lta/Spectral-RNN/blob/master/src/lorenz_data_generator.py
"""

import numpy as np
import torch


def generate_lorenz(
    length=2000,
    delta_t=0.01,
    sigma=10.0,
    beta=8.0 / 3.0,
    rho=28.0,
    batch_size=10,
    rnd=True,
    normalize=True,
):
    """
    Generate synthetic training data using the Lorenz system
    of equations (https://en.wikipedia.org/wiki/Lorenz_system):
    dxdt = sigma*(y - x)
    dydt = x * (rho - z) - y
    dzdt = x*y - beta*z
    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).
    Params:
        length = The number of timesteps. Total simulation time = length*delta_t
        delta_t: The step size.
        sigma: The first Lorenz parameter.
        beta: The second Lorenz parameter.
        rho: The thirs Lorenz parameter.
        batch_size: The first batch dimension.
        rnd: If true the lorenz seed is random.
    Returns:
        spikes: A Tensor of shape [batch_size, time, 1],
        states: A Tensor of shape [batch_size, time, 3].
    """

    # multi-dimensional data.
    def lorenz(x, t):
        return torch.stack(
            [
                sigma * (x[:, 1] - x[:, 0]),
                x[:, 0] * (rho - x[:, 2]) - x[:, 1],
                x[:, 0] * x[:, 1] - beta * x[:, 2],
            ],
            axis=1,
        )

    state0 = torch.tensor([8.0, 6.0, 30.0])
    state0 = torch.stack(batch_size * [state0], axis=0)
    if rnd:
        # print('Lorenz initial state is random.')
        state0 += (torch.rand([batch_size, 3]) - 0.5) * 8
    else:
        add_lst = []
        for i in range(batch_size):
            add_lst.append([0, float(i) * (1.0 / batch_size), 0])
        add_tensor = torch.stack(add_lst, axis=0)
        state0 += add_tensor
    states = [state0]
    for _ in range(length):
        states.append(states[-1] + delta_t * lorenz(states[-1], None))
    states = torch.stack(states, axis=1)
    spikes = torch.unsqueeze(torch.square(states[:, :, 0]), -1)
    # normalize
    if normalize:
        states = (states - torch.mean(states)) / torch.std(states)
        spikes = (spikes - torch.mean(spikes)) / torch.std(spikes)
    return spikes, states


def generate_mackey(batch_size=100, tmax=200, delta_t=1.0, rnd=True, device="cuda"):
    """
    Generate synthetic training data using the Mackey system
    of equations (http://www.scholarpedia.org/article/Mackey-Glass_equation):
    dx/dt = beta*(x'/(1+x'))
    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).
    Returns:
        spikes: A Tensor of shape [batch_size, time, 1],
    """
    steps = int(tmax / delta_t) + 200

    # multi-dimensional data.
    def mackey(x, tau, gamma=0.1, beta=0.2, n=10):
        return beta * x[:, -tau] / (1 + torch.pow(x[:, -tau], n)) - gamma * x[:, -1]

    tau = int(17 * (1 / delta_t))
    x0 = torch.ones([tau], device=device)
    x0 = torch.stack(batch_size * [x0], dim=0)
    if rnd:
        # print('Mackey initial state is random.')
        x0 += torch.empty(x0.shape, device=device).uniform_(-0.1, 0.1)
    else:
        x0 += [-0.01, 0.02]

    x = x0
    # forward_euler
    for _ in range(steps):
        res = torch.unsqueeze(x[:, -1] + delta_t * mackey(x, tau), -1)
        x = torch.cat([x, res], -1)
    discard = 200 + tau
    return x[:, discard:]


def blockify(data, block_length=20, block_prob=0.2):
    """Blockify the input data series by replacing blocks in the output with its mean."""
    steps = data.shape[-1] // block_length
    block_signal = []
    is_anomaly = np.zeros(steps * block_length, dtype=bool)
    for block_no in range(steps):
        random_no = np.random.uniform()
        start = block_no * block_length
        stop = (block_no + 1) * block_length
        if random_no < block_prob:
            block_mean = torch.mean(data[start:stop], dim=-1)
            block = block_mean * torch.ones([block_length], device=data.device)
            block_signal.append(block)
            is_anomaly[start:stop] = True
        else:
            block_signal.append(data[start:stop])
    return torch.cat(block_signal, -1), is_anomaly
