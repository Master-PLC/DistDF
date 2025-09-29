import math
import numpy as np
from numba import jit
from functools import lru_cache

import torch


# -----------------------------------------------------------------
# 1. Numba helpers  (全部 cache=True，编译结果持久化)
# -----------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def my_max(x, gamma):
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True, fastmath=True, cache=True)
def my_min(x, gamma):
    min_x, prob = my_max(-x, gamma)
    return -min_x, prob


@jit(nopython=True, fastmath=True, cache=True)
def my_max_hessian_product(p, z, gamma):
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True, fastmath=True, cache=True)
def my_min_hessian_product(p, z, gamma):
    return -my_max_hessian_product(p, z, gamma)


# -----------------------------------------------------------------
# 2. DTW 前向 / Hessian‑vec
# -----------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def dtw_grad(theta, gamma):
    m, n = theta.shape
    V = np.full((m + 1, n + 1), 1e10)
    V[0, 0] = 0.0
    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            v, Q[i, j] = my_min(
                np.array((V[i, j - 1], V[i - 1, j - 1], V[i - 1, j])), gamma
            )
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, n + 1] = 1.0
    Q[m + 1, n + 1] = 1.0
    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = (
                Q[i, j + 1, 0] * E[i, j + 1]
                + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j, 2] * E[i + 1, j]
            )

    return V[m, n], E[1 : m + 1, 1 : n + 1], Q, E


@jit(nopython=True, fastmath=True, cache=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m, n = Z.shape
    Vd = np.zeros((m + 1, n + 1))
    Qd = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            Vd[i, j] = (
                Z[i - 1, j - 1]
                + Q[i, j, 0] * Vd[i, j - 1]
                + Q[i, j, 1] * Vd[i - 1, j - 1]
                + Q[i, j, 2] * Vd[i - 1, j]
            )
            v = np.array((Vd[i, j - 1], Vd[i - 1, j - 1], Vd[i - 1, j]))
            Qd[i, j] = my_min_hessian_product(Q[i, j], v, gamma)

    Ed = np.zeros((m + 2, n + 2))
    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            Ed[i, j] = (
                Qd[i, j + 1, 0] * E[i, j + 1]
                + Q[i, j + 1, 0] * Ed[i, j + 1]
                + Qd[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j + 1, 1] * Ed[i + 1, j + 1]
                + Qd[i + 1, j, 2] * E[i + 1, j]
                + Q[i + 1, j, 2] * Ed[i + 1, j]
            )

    return Vd[m, n], Ed[1 : m + 1, 1 : n + 1]


# -----------------------------------------------------------------
# 3. PathDTWBatch（Python 侧逻辑与之前一致）
# -----------------------------------------------------------------
class PathDTWBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D, gamma):
        B, N, _ = D.shape
        dev = D.device
        D_np = D.cpu().numpy()
        g = float(gamma)

        grad_gpu = torch.empty((B, N, N), device=dev)
        Q_gpu = torch.empty((B, N + 2, N + 2, 3), device=dev)
        E_gpu = torch.empty((B, N + 2, N + 2), device=dev)

        for k in range(B):
            _, gk, Qk, Ek = dtw_grad(D_np[k], g)
            grad_gpu[k] = torch.from_numpy(gk).to(dev)
            Q_gpu[k] = torch.from_numpy(Qk).to(dev)
            E_gpu[k] = torch.from_numpy(Ek).to(dev)

        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, torch.tensor(g))
        return grad_gpu.mean(0)

    @staticmethod
    def backward(ctx, grad_out):
        grad_gpu, D, Qg, Eg, g = ctx.saved_tensors
        B, N, _ = D.shape
        dev = grad_out.device
        g = float(g)
        Z = grad_out.cpu().numpy()
        D_np = D.cpu().numpy()
        Q_np = Qg.cpu().numpy()
        E_np = Eg.cpu().numpy()

        H = torch.empty_like(D)
        for k in range(B):
            _, hk = dtw_hessian_prod(D_np[k], Z, Q_np[k], E_np[k], g)
            H[k] = torch.from_numpy(hk).to(dev)

        return H, None


# -----------------------------------------------------------------
# 4. Soft‑DTW（与旧版相同，也可加 @jit(..., cache=True)）
# -----------------------------------------------------------------
@jit(nopython=True, cache=True)
def compute_softdtw(D, gamma):
    N, M = D.shape
    R = np.full((N + 2, M + 2), 1e8)
    R[0, 0] = 0.0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0, r1, r2 = -R[i - 1, j - 1] / gamma, -R[i - 1, j] / gamma, -R[i, j - 1] / gamma
            rmax = max(r0, r1, r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            R[i, j] = D[i - 1, j - 1] - gamma * (np.log(rsum) + rmax)
    return R


@jit(nopython=True, cache=True)
def compute_softdtw_backward(D_, R, gamma):
    N, M = D_.shape
    D = np.zeros((N + 2, M + 2))
    E = np.zeros_like(D)
    D[1:N + 1, 1:M + 1] = D_
    E[-1, -1] = 1
    R[:, -1] = R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a = math.exp((R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma)
            b = math.exp((R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma)
            c = math.exp((R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma)
            E[i, j] = a * E[i + 1, j] + b * E[i, j + 1] + c * E[i + 1, j + 1]
    return E[1:N + 1, 1:M + 1]


class SoftDTWBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D, gamma=1.0):
        B, N, _ = D.shape
        dev = D.device
        g = float(gamma)
        D_np = D.cpu().numpy()

        R_all = torch.empty((B, N + 2, N + 2), device=dev)
        loss = 0.0
        for k in range(B):
            Rk = torch.from_numpy(compute_softdtw(D_np[k], g)).to(dev)
            R_all[k] = Rk
            loss += Rk[-2, -2]
        ctx.save_for_backward(D, R_all, torch.tensor(g))
        return loss / B

    @staticmethod
    def backward(ctx, grad_out):
        D, R_all, g = ctx.saved_tensors
        B, N, _ = D.shape
        dev = grad_out.device
        g = float(g)
        D_np = D.cpu().numpy()
        grads = torch.empty_like(D)
        for k in range(B):
            grads[k] = torch.from_numpy(compute_softdtw_backward(D_np[k], R_all[k].cpu().numpy(), g)).to(dev)
        return grad_out * grads, None


# -----------------------------------------------------------------
# 5. Ω 缓存：LRU
# -----------------------------------------------------------------
@lru_cache(maxsize=None)
def _omega_cached(N: int, device_str: str):
    device = torch.device(device_str)
    idx = torch.arange(1, N + 1, device=device).view(N, 1).float()
    return torch.cdist(idx, idx, p=1).pow(2) / (N * N)


# -----------------------------------------------------------------
# 6. 其它工具
# -----------------------------------------------------------------
def pairwise_distances(x, y=None):
    if y is None:
        y = x
    return torch.cdist(x, y).pow(2)


# -----------------------------------------------------------------
# 7. Public API
# -----------------------------------------------------------------
def dilate_loss(outputs, targets, alpha, gamma, device):
    B, N, _ = outputs.shape
    D = torch.cdist(targets.view(B, N, -1), outputs.view(B, N, -1)).pow(2)

    loss_shape = SoftDTWBatch.apply(D, gamma)

    path = PathDTWBatch.apply(D, gamma)
    Omega = _omega_cached(N, str(device))
    loss_temporal = (path * Omega).sum()

    total = alpha * loss_shape + (1 - alpha) * loss_temporal
    return total, loss_shape, loss_temporal