import torch


def compute_R(z, neighbors, h, beta, M, p):
    """
    Compute R(z) = h^2 ∑ φ(g_i) using PyTorch autograd.

    Parameters:
        z : tensor of shape (N, d)  # d=channels (e.g., 3 wavelet bands)
        neighbors : list of index tensors, each giving indices in N(i)
        h, beta, M, p : scalars

    Returns:
        scalar tensor R(z)
    """
    # Step 1: magnitude of each vector coefficient ‖z_k‖
    m = torch.norm(z, dim=1)  # shape (N,)

    # Step 2: compute g_i for each i
    g_list = []
    for N_i in neighbors:
        g_i = torch.norm(m[N_i], p=p)  # ( ∑ |m_j|^p )^{1/p}
        g_list.append(g_i)

    g = torch.stack(g_list)  # shape (#i,)

    # Step 3: φ_{β,M}(t) = M - (1/β) * log(1 + exp(β(t - M)))
    phi = M - (1 / beta) * torch.log(1 + torch.exp(beta * (g - M)))

    # Step 4: R(z) = h² * Σ φ(g_i)
    return h ** 2 * phi.sum()


# Example input
N, d = 50, 3
z = torch.randn(N, d, requires_grad=True)

# Example neighborhoods: 5-point windows
neighbors = [torch.tensor([i, i+1]) for i in range(N-1)]
neighbors.append(torch.tensor([N-1]))  # last one fix

# parameters
h = 0.1
beta = 5
M = 1.0
p = 8

# Compute R(z) and gradient
R = compute_R(z, neighbors, h, beta, M, p)
R.backward()

print("R(z):", R.item())
print("Gradient shape:", z.grad.shape)
print("First few grads:\n", z.grad[:5])
