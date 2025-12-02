import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('tkagg')
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_config():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--M', type=float, default=0.3, help='M')
    parser.add_argument('--p', type=float, default=7, help='p')
    parser.add_argument('--neighborhood_size', type=int, default=3, help='neighborhood_size')
    parser.add_argument('--lamb', type=float, default=1, help='lamb')
    return parser.parse_args()



class LocalLPRegularizer(nn.Module):
    """
    z: (B, 3, H, W) = (subbands, spatial)
    """
    def __init__(self, p=8, beta=5.0, M=1.0, h=0.5, neighborhood_size=3):
        super().__init__()
        self.p = float(p)
        self.beta = float(beta)
        self.M = float(M)
        self.h = float(h)
        self.neighborhood_size = float(neighborhood_size)

        if isinstance(neighborhood_size, int):
            k_h = k_w = neighborhood_size
        else:
            k_h, k_w = neighborhood_size

        weight = torch.ones(1, 1, k_h, k_w)
        self.register_buffer("weight", weight)
        self.pad_h = k_h // 2
        self.pad_w = k_w // 2

    def phi(self, t):
        # NOTE: this φ can go negative for large t (you already saw that)
        return self.M - (1.0 / self.beta) * torch.log1p(
            torch.exp(self.beta * (self.M-t))
        ) - (self.M - (1.0 / self.beta) * torch.log1p(
            torch.exp(self.beta * (self.M-t*0))
        ))

    def forward(self, z):
        """
        z: (B, 3, H, W)
        """
        # |z_j| = L2 across subbands → (B, H, W)
        mag = torch.norm(z, dim=1)

        # |z_j|^p and add channel dim for conv2d → (B,1,H,W)
        mag_p = mag.pow(self.p).unsqueeze(1)
        # neighborhood sum → (B,1,H,W)
        neigh_sum = F.conv2d(
            mag_p,
            self.weight,
            padding=(self.pad_h, self.pad_w)
        )
        # print(neigh_sum)
        # g_i = (...)^(1/p) → (B,H,W)
        g = neigh_sum.pow(1.0 / self.p).squeeze(1)
        # print(g)
        # φ(g) → (B,H,W)
        phi_vals = self.phi(g)

        # scalar regularizer
        R = (self.h ** 2) * phi_vals.sum()
        return R


class WaveletLPRegularizer(nn.Module):
    """
    u: (B,1,H,W)  → DWT → z (B,3,H',W') → LocalLPRegularizer
    """
    def __init__(self,
                 wave='bior4.4',
                 level=1,
                 p=8,
                 beta=5.0,
                 M=1.0,
                 h=0.5,
                 neighborhood_size=3):
        super().__init__()
        self.dwt = DWTForward(J=level, wave=wave, mode='symmetric')
        self.reg = LocalLPRegularizer(
            p=p, beta=beta, M=M, h=h, neighborhood_size=neighborhood_size
        )

    def forward(self, u):
        yl, yh_list = self.dwt(u)

        # level-1 highpass coefficients: (B, C_in, 3, H', W')
        z = yh_list[0]

        # assume grayscale input → C_in=1, squeeze it
        z = z.squeeze(1)       # (B,3,H',W')

        return self.reg(z)

config = get_config()
p = config.p
M = config.M
neighborhood_size = config.neighborhood_size
lamb = config.lamb
u = plt.imread('dog.png')
B, H, W = 1, u.shape[0], u.shape[1]
original = torch.tensor(u).unsqueeze(0).unsqueeze(0)
noise = torch.randn_like(original)
u = noise*.2 + original
u.requires_grad_(True)
f = u.clone()
R_module = WaveletLPRegularizer(
    wave='db2',
    level=1,
    p=p,
    beta=50,
    M=M,
    h=0.5,
    neighborhood_size=neighborhood_size
)

R = R_module(u)     # scalar
R.backward()
starting_r = float(R)
optimizer = torch.optim.SGD([u], lr=5e-4)

for it in tqdm(range(500), desc="Optimizing u"):
    optimizer.zero_grad()

    R = R_module(u)       # your wavelet regularizer
    data_loss = 0.5 * ((u - f)**2).sum()   # example data term
    # if it % 50 == 0:
    #     print("\ndata_loss", data_loss)
    #     print("R", R)
    loss = data_loss + lamb * neighborhood_size**2 * R
    loss.backward()

    optimizer.step()
    # print(it, float(loss))
print("Starting R =", starting_r)
print("Ending R =", float(R))
print("u.grad mean abs =", u.grad.abs().mean())
original_np = original.detach().cpu().numpy().squeeze()
f_np        = f.detach().cpu().clamp(0,1).numpy().squeeze()
u_np        = u.detach().cpu().clamp(0,1).numpy().squeeze()
print(f_np.mean(), u_np.mean())
# Compute PSNR
original_psnr = psnr(original_np, f_np, data_range=1.0)
new_psnr = psnr(original_np, u_np, data_range=1.0)

print("PSNR(original vs f):", original_psnr)
print("PSNR(original vs u):", new_psnr)

plt.figure(figsize=(16, 10))

# Original (clean)
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original.detach().cpu().numpy()[0, 0], cmap='gray')
plt.axis('off')

# Noisy (f)
plt.subplot(1, 3, 2)
plt.title("Noisy (f)")
plt.imshow(f_np, cmap='gray')
plt.axis('off')

# Reconstructed (u)
plt.subplot(1, 3, 3)
plt.title("Reconstructed (u)")
plt.imshow(u_np, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


