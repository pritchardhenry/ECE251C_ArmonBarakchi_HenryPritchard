import numpy as np
import dtcwt
import pywt

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




def WT(u, wavelet='bior4.4'):
    """
    Apply PyWavelets 2D transform (single level) using bior4.4.
    Returns detail coefficients stacked as z: (3, H, W).
    """
    coeffs = pywt.wavedec2(u, wavelet=wavelet, level=1)
    cA, (cH, cV, cD) = coeffs

    # Stack only detail coefficients
    z = np.stack([cH, cV, cD])  # shape = (3, H, W)
    return z, coeffs






def compute_R(u, h=0.5, beta=5, M=1, p=8, r=3, wavelet='bior4.4'):
    z, coeffs = WT(u, wavelet)   # z: (3, H, W)
    m = np.sqrt(np.sum(z**2, axis=0)) # (H, W)

    H, W = m.shape
    R_total = 0.

    for i in range(H):
        for j in range(W):
            g_ij = local_p_norm(m, i, j, p=p, r=r)
            R_total += phi_beta_M(g_ij, beta=beta, M=M)

    return h**2 * R_total


def compute_grad_R(u, beta=5, M=0.2, p=8, r=2, h=0.5, wavelet='bior4.4'):
    """
    Compute ∇R/∇u using PyWavelets (real-valued).
    """
    z, coeffs = WT(u, wavelet)  # z: (3, H, W)
    K, H_, W_ = z.shape

    # Magnitude across channels
    m = np.sqrt(np.sum(z**2, axis=0)) + 1e-8

    # Precompute g(i,j)
    g_matrix = np.zeros_like(m)
    for i in range(H_):
        for j in range(W_):
            g_matrix[i, j] = local_p_norm(m, i, j, p=p, r=r)

    # Allocate gradient in wavelet domain
    grad_R_z = np.zeros_like(z)

    # Loop through all pixels
    for k1 in range(H_):
        for k2 in range(W_):

            if m[k1, k2] == 0:
                continue

            for i in range(max(0, k1-r), min(H_, k1+r+1)):
                for j in range(max(0, k2-r), min(W_, k2+r+1)):
                    g_ij = g_matrix[i, j]

                    # φ'(g)
                    phi_prime = -1 / (1 + np.exp(-beta*(g_ij - M)))

                    # ∂g/∂z
                    dz = (np.abs(z[:,k1,k2])**(p-2)) * z[:,k1,k2] * (g_ij**(1-p))

                    grad_R_z[:,k1,k2] += h**2 * phi_prime * dz

    # Map back to image domain using inverse wavelet transform
    # Replace only the detail subbands; keep original approximation
    grad_coeffs = [coeffs[0], tuple(grad_R_z)]
    grad_R_u = pywt.waverec2(grad_coeffs, wavelet)

    return grad_R_u
