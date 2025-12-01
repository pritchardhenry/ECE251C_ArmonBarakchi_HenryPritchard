import numpy as np
import dtcwt

def phi_beta_M(t, beta=5, M=0.2):
    x = beta * (t - M)
    # Numerically stable softplus:
    softplus = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    return M - (1 / beta) * softplus



def channel_l2(z):
    """
    Compute per-pixel L2 norm across channels/orientations.
    z shape: (K, H, W) → returns (H, W)
    """
    return np.sqrt(np.sum(np.abs(z) ** 2, axis=0))


def local_p_norm(m, i, j, p=8, r=2):
    """
    Compute g_{i,p}(z): p-norm of magnitudes in an r-radius neighborhood.
    m is (H, W) array of magnitudes.
    """
    H, W = m.shape
    row_start = max(i - r, 0)
    row_end = min(i + r, H - 1)
    col_start = max(j - r, 0)
    col_end = min(j + r, W - 1)

    local = m[row_start:row_end + 1, col_start:col_end + 1]
    return np.sum(local ** p) ** (1 / p)

def WT(u):
    """
       Apply DTCWT and return ONLY the finest-scale highpass coefficients.
       """
    transform = dtcwt.Transform2d()
    coeffs = transform.forward(u)

    # Finest scale = coeffs.highpasses[0], shape (6, H/2, W/2)
    z = coeffs.highpasses[0]

    return z, coeffs, transform





def compute_R(z, h=0.5, beta=5, M=1, p=8, r=3):
    """
    Compute R(z) = h^2 ∑ φ_{β,M}( g_{i,p}(z) ).

    Parameters:
        z: wavelet coefficients, shape (K, H, W)
        h: scaling constant (pixel size or grid spacing)
        p: exponent for neighborhood p-norm
        r: neighborhood radius (box window)
    """


    # --- Step 2: compute per-pixel L2 magnitude of z across orientations ---
    m = np.sqrt(np.sum(np.abs(z) ** 2, axis=0))  # shape (H, W)

    H, W = m.shape
    R_total = 0.0

    # Step 2: loop over pixels (i,j) and compute R
    for i in range(H):
        for j in range(W):
            g_ij = local_p_norm(m, i, j, p=p, r=r)
            R_total += phi_beta_M(g_ij, beta=beta, M=M)

    return h ** 2 * R_total


def compute_grad_R(z, beta=5, M=0.2, p=8, r=2, h=0.5):
    """
    Compute ∇R/∇z where R(z) = h^2 * sum φ(g_i(z)),
    and z has shape (K, H, W) from DTCWT.
    """
    K, H, W = z.shape

    # Magnitude of z across channels/orientations
    m = np.sqrt(np.sum(np.abs(z)**2, axis=0))   # (H, W)

    # Store g_i values
    g_matrix = np.zeros((H, W))

    # Step 1: compute g_i for all pixels
    for i in range(H):
        for j in range(W):
            g_matrix[i, j] = local_p_norm(m, i, j, p=p, r=r)

    # Step 2: Compute gradient ∂R/∂z
    grad_R_z = np.zeros_like(z, dtype=np.complex128)

    for k1 in range(H):
        for k2 in range(W):

            # Skip pixels where ||z||=0
            if m[k1, k2] == 0:
                continue

            for i in range(max(k1-r,0), min(k1+r+1, H)):
                for j in range(max(k2-r,0), min(k2+r+1, W)):

                    g_ij = g_matrix[i, j]

                    # Derivative φ'(g) = -σ(β(g - M))
                    phi_prime = -1 / (1 + np.exp(-beta*(g_ij - M)))

                    # ∂g/∂z = |z|^{p-2} * z * g^{1-p}
                    dz = (np.abs(z[:, k1, k2])**(p-2)) * z[:, k1, k2] * (g_ij**(1-p))

                    grad_R_z[:, k1, k2] += h**2 * phi_prime * dz

    return grad_R_z
