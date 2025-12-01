import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Regularizer import compute_R
from Regularizer import compute_grad_R
from Regularizer import WT
import dtcwt
import pywt
#note: if you want to leave the identity case (denoising). Implement A_forward, A_adjoint
#and uncomment the lines from data_fidelity and data_fidelity_grad
def A_forward(u):
    return u  # identity operator

def A_adjoint(v):
    return v  # identity adjoint
def data_fidelity(u, f):
    """
    Computes ||A*u - f||^2
    """
    #Au = A_forward(u)
    return np.sum((u - f)**2)


def data_fidelity_grad(u, f, A_forward = None, A_adjoint = None):
    """
    Computes gradient of ||A*u - f||^2 = 2 * A^T(Au - f)
    """
    #Au = A_forward(u)
    #return 2 * A_adjoint(Au - f)
    return 2 * (u - f)

def W(u):
    return 0

def run_reconstruction(f, lam=0.75, lr=1.5e-2, h=0.5, beta=5, M=0.2, p=4, r = 4, max_iter=50):
    u = f.copy()

    for it in range(max_iter):

        if it % 10 == 0 or it == max_iter - 1:
            print(f"Iter {it:4d} | Data: {data_fidelity(u,f):.4f} | R(u): {compute_R(u, h, beta, M, p, r):.4f}")


        # Now safely apply inverse (adjoint mapping back to image space)
        grad_R_u = compute_grad_R(u, beta, M, p, r, h)  # 2) gradient in wavelet domain
        grad_R_u = grad_R_u[:u.shape[0], :u.shape[1]]
        grad_F = data_fidelity_grad(u,f) + lam * grad_R_u  # full gradient

        # Update
        u = u - lr * grad_F

        # Keep u in valid image range
        u = np.clip(u, 0, 1)

    return u

# ============================================================
# 1️⃣ Load Image and Normalize
# ============================================================
image_path = "/Users/armonbarakchi/Desktop/ECE251C_ArmonBarakchi_HenryPritchard/dog.png"
img = Image.open(image_path).convert("L")  # Grayscale
f = np.array(img) / 255.0                 # Clean, normalized in [0,1]

# ============================================================
# 2️⃣ Add Gaussian Noise
# ============================================================
noise_std = 0.2
noise = noise_std * np.random.randn(*f.shape)
f_noisy = np.clip(f + noise, 0, 1)

# ============================================================
# 3️⃣ Run Reconstruction
# ============================================================
u_rec = run_reconstruction(f_noisy)   # Must return normalized uint8 or float [0,1]

# ============================================================
# 4️⃣ Display Clean, Noisy, and Reconstructed
# ============================================================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(f, cmap='gray')
plt.title("Original Clean")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(f_noisy, cmap='gray')
plt.title("Noisy Input")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(u_rec, cmap='gray')
plt.title("Reconstructed")
plt.axis('off')

plt.show()
