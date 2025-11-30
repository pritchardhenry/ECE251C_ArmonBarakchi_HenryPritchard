import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_wavelets import DTCWTForward, DTCWTInverse  # PyTorch-native DTCWT

# ============================================================
# 1️⃣ Custom Regularizer (same as your autograd-based version)
# ============================================================
def compute_R(z, neighbors, h, beta, M, p):
    m = torch.norm(z, dim=1)  # magnitude per coefficient group

    g_list = []
    for N_i in neighbors:
        g_i = torch.norm(m[N_i], p=p)
        g_list.append(g_i)

    g = torch.stack(g_list)

    phi = M - (1 / beta) * torch.log(1 + torch.exp(beta * (g - M)))
    return h**2 * phi.sum()

# ============================================================
# 2️⃣ Dual-Tree Wavelet Transform (DTCWT) – Fully Differentiable
# ============================================================
dt_forward = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')   # 3 levels
dt_inverse = DTCWTInverse()

def W(u):
    yl, yh = dt_forward(u)  # yh is a list of tensors, each: [1,1,H,W,6]
    bands = []

    for level in yh:
        # Directly flatten all orientation channels
        bands.append(level.reshape(-1))

    return torch.cat(bands).reshape(-1,1)  # (N,1)

# ============================================================
# 3️⃣ Main Reconstruction Loop (Adam)
# ============================================================
def run_reconstruction(f, lam=0.5, lr=1e-2, h=0.5, beta=5, M=1.0, p=4, max_iter=200):
    """Gradient-based image restoration with your regularizer."""
    u = (f + 0.05 * torch.randn_like(f)).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=lr)

    # Build neighbors dynamically after first W(u)
    with torch.no_grad():
        z_init = W(u)
    N = z_init.shape[0]
    neighbors = [torch.arange(max(0, i-2), min(N, i+3)) for i in range(N)]

    for it in range(max_iter):
        optimizer.zero_grad()

        data = 0.5 * torch.norm(u - f)**2

        z = W(u)
        R_u = compute_R(z, neighbors, h, beta, M, p)

        loss = data + lam * R_u
        loss.backward()
        optimizer.step()

        if it % 20 == 0 or it == max_iter - 1:
            print(f"Iter {it:3d} | Loss={loss.item():.4f} | Data={data.item():.4f} | R={R_u.item():.4f}")

    return u.detach()

# ============================================================
# 4️⃣ Load Real Image (SET YOUR IMAGE PATH HERE)
# ============================================================
image_path = "/Users/armonbarakchi/Desktop/ECE251C_ArmonBarakchi_HenryPritchard/dog.png"  # 🔴 Change this

img = Image.open(image_path).convert("L")
img = img.resize((128,128))

f_np = np.array(img)/255.0
f = torch.tensor(f_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Optional: add synthetic Gaussian noise
noise = 0.1 * torch.randn_like(f)
f_noisy = f + noise

plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.imshow(f.squeeze(), cmap="gray"); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(f_noisy.squeeze(), cmap="gray"); plt.title("Noisy")
plt.show()

# ============================================================
# 5️⃣ Run Reconstruction
# ============================================================
u_rec = run_reconstruction(f_noisy)

# ============================================================
# 6️⃣ Display Final Result
# ============================================================
plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.imshow(f_noisy.squeeze(), cmap="gray"); plt.title("Noisy Input")
plt.subplot(1,2,2); plt.imshow(u_rec.squeeze(), cmap="gray"); plt.title("Reconstructed")
plt.show()
