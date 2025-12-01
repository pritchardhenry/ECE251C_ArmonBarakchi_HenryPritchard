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
            torch.exp(self.beta * (t - self.M))
        )

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

        # g_i = (...)^(1/p) → (B,H,W)
        g = neigh_sum.pow(1.0 / self.p).squeeze(1)

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



u = plt.imread('dog.png')*256
B, H, W = 1, u.shape[0], u.shape[1]
u = torch.tensor(u).unsqueeze(0).unsqueeze(0)
noise = torch.randn_like(u)
u = noise*40 + u
u.requires_grad_(True)
f = u.clone()
R_module = WaveletLPRegularizer(
    wave='haar',
    level=1,
    p=4,
    beta=5,
    M=10,
    h=0.5,
    neighborhood_size=2
)

R = R_module(u)     # scalar
R.backward()

optimizer = torch.optim.SGD([u], lr=1e-4)

for it in tqdm(range(10), desc="Optimizing u"):
    optimizer.zero_grad()

    R = R_module(u)       # your wavelet regularizer
    data_loss = 0.5 * ((u - f)**2).sum()   # example data term

    loss = data_loss + R
    loss.backward()

    optimizer.step()
    # print(it, float(loss))

print("R =", float(R))
print("u.grad mean abs =", u.grad.abs().mean())

print(u)
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1)
plt.imshow(f.detach().numpy()[0][0], cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(u.detach().numpy()[0][0], cmap='gray')
plt.show()

