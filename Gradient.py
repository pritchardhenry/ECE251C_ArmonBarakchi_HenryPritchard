import numpy as np


def phi_prime(t, beta, M):
    return -np.exp(beta * (t - M)) / (1 + np.exp(beta * (t - M)))


def grad_R(z, neighbors, h, beta, M, p):
    """
    Compute ∇R(z) using explicit ∂g_i/∂z_k.

    Parameters:
        z: 1D numpy array of coefficients
        neighbors: list of arrays, neighbors[i] gives indices in N(i)
        h, beta, M, p: scalar parameters
    """
    N = len(z)
    grad = np.zeros_like(z, dtype=float)

    # Step 1: Compute g_{i,p}(z) for each i
    g = np.zeros(N)
    for i in range(N):
        N_i = neighbors[i]
        g[i] = (np.sum(np.abs(z[N_i]) ** p)) ** (1 / p)

    # Step 2: Compute φ'(g_i)
    phi_p = phi_prime(g, beta, M)

    # Step 3: Accumulate gradient
    for k in range(N):  # loop over variable entries (z_k)
        for i, N_i in enumerate(neighbors):  # loop over neighborhoods
            if k in N_i and g[i] != 0:  # ∂g/∂z_k ≠ 0 only if k ∈ N(i)
                d_g = (np.abs(z[k]) ** (p - 1)) * np.sign(z[k]) * (g[i] ** (1 - p))
                grad[k] += phi_p[i] * d_g

    return h ** 2 * grad

