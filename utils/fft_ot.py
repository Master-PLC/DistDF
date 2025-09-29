from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from ot.backend import get_backend
from ot.utils import dots, list_to_array


def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian RBF kernel between x and y"""
    # x: [B, D], y: [B, D]
    diff = x[:, None, :] - y[None, :, :]  # [B, B, D]
    dist_sq = (diff ** 2).sum(dim=-1)  # [B, B]
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def polynomial_kernel(x, y, degree=3, gamma=1.0, coef0=1.0):
    """Compute polynomial kernel between x and y"""
    # x: [B, D], y: [B, D] 
    dot_product = torch.mm(x, y.T)  # [B, B]
    return (gamma * dot_product + coef0) ** degree


def linear_kernel(x, y):
    """Compute linear kernel between x and y"""
    return torch.mm(x, y.T)


def compute_mmd(X, Y, kernel_func, **kernel_params):
    """Compute MMD between two sets of samples"""
    # X: [B, D], Y: [B, D]
    XX = kernel_func(X, X, **kernel_params).mean()
    YY = kernel_func(Y, Y, **kernel_params).mean()  
    XY = kernel_func(X, Y, **kernel_params).mean()
    
    mmd = XX - 2 * XY + YY
    return mmd


def multi_kernel_mmd(X, Y, sigmas=[0.1, 1.0, 10.0]):
    """Compute MMD with multiple Gaussian kernels"""
    mmd = 0
    for sigma in sigmas:
        mmd += compute_mmd(X, Y, gaussian_kernel, sigma=sigma)
    return mmd / len(sigmas)


def cal_wasserstein(X1, X2, distance, ot_type='sinkhorn', normalize=1, mask_factor=0.01, numItermax=10000, stopThr=1e-4, reg_sk=0.1, reg_m=10, var_weight=1.0, mean_weight=1.0, eps=1e-8):
    """
    X1: prediction sequence with shape [B, T, D]
    X2: label sequence with shape [B, T, D]
    distance: the method to calculate the pair-wise sample distance
    ot_type: the definition of OT problem
    reg_sk: the strength of entropy regularization in Sinkhorn
    reg_m: the strength of mass-preservation constraint in UOT
    Currently we can fix the ot_type as sinkhorn, and there are key parameters to tune:
        1. whether normalize is needed?
        2. which distance is better? time/fft_2norm/fft_1norm, other distances do not need to investigate now.
        3. the optimum reg_sk? Possible range: [0.01-5]
    Moreover, the weight of the loss should be tuned in this duration. Try both relative weights (\alpha for wass, 1-\alpha for mse) and absolute weights (\alpha for wass, 1 for mse)
    """
    B, T, D = X1.shape
    device = X1.device

    if distance == 'time':
        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
    elif distance == 'fft_2norm':
        X1 = torch.fft.rfft(X1.transpose(1, 2))
        X2 = torch.fft.rfft(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()**2).sum(-1)
    elif distance == 'fft_1norm':
        X1 = torch.fft.rfft(X1.transpose(1, 2))
        X2 = torch.fft.rfft(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()).sum(-1)
    elif distance == 'fft_2norm_2d':
        X1 = torch.fft.rfft2(X1.transpose(1, 2))  # [B, D, T//2+1]
        X2 = torch.fft.rfft2(X2.transpose(1, 2))  # [B, D, T//2+1]
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()**2).sum(-1)
    elif distance == 'fft_1norm_2d':
        X1 = torch.fft.rfft2(X1.transpose(1, 2))
        X2 = torch.fft.rfft2(X2.transpose(1, 2))
        M = ((X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:]).abs()).sum(-1)
    elif distance == 'dct_2norm':
        X1 = dct.dct(X1.transpose(1, 2), norm='ortho')  # [B, D, T]
        X2 = dct.dct(X2.transpose(1, 2), norm='ortho')  # [B, D, T]
        M = ((X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :]) ** 2).sum(-1)
    elif distance == 'dct_1norm':
        X1 = dct.dct(X1.transpose(1, 2), norm='ortho')
        X2 = dct.dct(X2.transpose(1, 2), norm='ortho')
        M = (X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :]).abs().sum(-1)
    elif distance == 'dct_2norm_2d':
        X1 = dct.dct_2d(X1.transpose(1, 2), norm='ortho')  # [B, D, T]
        X2 = dct.dct_2d(X2.transpose(1, 2), norm='ortho')  # [B, D, T]
        M = ((X1.flatten(1)[:, None, :] - X2.flatten(1)[None, :, :]) ** 2).sum(-1)
    elif distance == 'fft_mag':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
        M = ot.dist(X1.flatten(1), X2.flatten(1), metric='sqeuclidean', p=2)
    elif distance == 'fft_mag_abs':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).abs()
        X2 = torch.fft.rfft(X2.transpose(1, 2)).abs()
        M = torch.norm(X1.flatten(1)[:,None,:] - X2.flatten(1)[None,:,:], p=1, dim=2)
    elif distance == 'fft_2norm_kernel':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)
        sigma = 1
        M = -1 * torch.exp(-1 * ((X1[:,None,:] - X2[None,:,:]).abs()).sum(-1) / 2/sigma)
    elif distance == 'fft_2norm_multikernel':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)
        w = [0.5, 0.5]
        sigma = [0.1, 1]
        dist_weighted = [-1 * w[i] * torch.exp(-1 * ((X1[:,None,:] - X2[None,:,:]).abs()).sum(-1) / 2/sigma[i]) for i in range(len(w))]
        M = sum(dist_weighted)
    elif distance == 'fft_2norm_per_dim':
        X1 = torch.fft.rfft(X1.transpose(1, 2))  # [B, D, T//2+1]
        X2 = torch.fft.rfft(X2.transpose(1, 2))  # [B, D, T//2+1]
        
        # Correct calculation of M
        M = ((X1[:, None, :, :] - X2[None, :, :, :]).abs()**2).sum(-1)  # [B, B, D]
        
        if normalize == 1:
            # equals to M = torch.stack([M[..., d] / M[..., d].max() for d in range(D)], dim=-1)
            M = M / M.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]

        loss = 0
        a, b = torch.ones((B,), device=device) / B, torch.ones((B,), device=device) / B

        for d in range(D):
            M_d = M[:, :, d]

            if mask_factor > 0:
                mask = torch.ones_like(M_d) - torch.eye(B, device=device)
                M_d = M_d + mask_factor * mask

            if ot_type == 'sinkhorn':
                pi = ot.sinkhorn(a, b, M_d, reg=reg_sk, numItermax=numItermax, stopThr=stopThr)
            elif ot_type == 'emd':
                pi = ot.emd(a, b, M_d, numItermax=numItermax)
            elif ot_type == 'uot':
                pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M_d, reg=reg_sk, stopThr=stopThr, numItermax=numItermax, reg_m=reg_m)
            elif ot_type == 'uot_mm':
                pi = ot.unbalanced.mm_unbalanced(a, b, M_d, reg_m=reg_m, numItermax=numItermax, stopThr=stopThr)

            loss += (pi * M_d).sum()

        loss = loss / D
        return loss

    elif distance == '2norm_per_dim':
        # Correct calculation of M
        M = ((X1[:, None, :, :] - X2[None, :, :, :])**2).sum(-1)  # [B, B, D]
        
        if normalize == 1:
            # equals to M = torch.stack([M[..., d] / M[..., d].max() for d in range(D)], dim=-1)
            M = M / M.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]

        loss = 0
        a, b = torch.ones((B,), device=device) / B, torch.ones((B,), device=device) / B

        for d in range(D):
            M_d = M[:, :, d]

            if mask_factor > 0:
                mask = torch.ones_like(M_d) - torch.eye(B, device=device)
                M_d = M_d + mask_factor * mask

            if ot_type == 'sinkhorn':
                pi = ot.sinkhorn(a, b, M_d, reg=reg_sk, numItermax=numItermax, stopThr=stopThr)
            elif ot_type == 'emd':
                pi = ot.emd(a, b, M_d, numItermax=numItermax)
            elif ot_type == 'uot':
                pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M_d, reg=reg_sk, stopThr=stopThr, numItermax=numItermax, reg_m=reg_m)
            elif ot_type == 'uot_mm':
                pi = ot.unbalanced.mm_unbalanced(a, b, M_d, reg_m=reg_m, numItermax=numItermax, stopThr=stopThr)

            loss += (pi * M_d).sum()

        loss = loss / D
        return loss

    elif distance == 'fft_wasserstein_1d':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)

        X1 = X1.abs()
        X2 = X2.abs()

        a, b = torch.ones((B, X1.shape[1]), device=device) / B, torch.ones((B, X2.shape[1]), device=device) / B
        loss = ot.lp.wasserstein_1d(X1, X2, a, b, p=2)
        loss = loss.mean()
        return loss

    elif distance == 'emd_per_dim':
        loss = 0
        a, b = torch.ones((B,), device=device) / B, torch.ones((B,), device=device) / B

        for d in range(D):
            M_d = ot.dist(X1[:, :, d], X2[:, :, d], metric='sqeuclidean', p=2)
            if normalize == 1:
                M_d = M_d / M_d.max()

            if mask_factor > 0:
                mask = torch.ones_like(M_d) - torch.eye(B, device=device)
                M_d = M_d + mask_factor * mask

            loss += ot.emd2(a, b, M_d, numItermax=numItermax)

        loss = loss / D
        return loss

    elif distance == 'wasserstein_1d_per_dim':
        a, b = torch.ones((B, T), device=device) / B, torch.ones((B, T), device=device) / B

        loss = 0
        for d in range(D):
            X1_d = X1[..., d]
            X2_d = X2[..., d]
            loss += ot.lp.wasserstein_1d(X1_d, X2_d, a, b, p=2)
        loss = loss / D
        loss = loss.mean()
        return loss

    elif distance == 'fft_wasserstein_empirical':
        X1 = torch.fft.rfft(X1.transpose(1, 2)).flatten(1)
        X2 = torch.fft.rfft(X2.transpose(1, 2)).flatten(1)

        X1 = X1.abs()
        X2 = X2.abs()

        a, b = torch.ones((B,1), device=device, dtype=X1.dtype) / B, torch.ones((B,1), device=device, dtype=X2.dtype) / B
        # loss = ot.gaussian.empirical_bures_wasserstein_distance(X1, X2, reg=reg_sk, ws=a, wt=b)
        loss = empirical_bures_wasserstein_distance(X1, X2, reg=reg_sk, ws=a, wt=b)
        loss = loss.mean()
        return loss

    elif distance == 'wasserstein_empirical_per_dim':
        #! 去掉rfft, 对each dimension考虑
        a, b = torch.ones((B,1), device=device, dtype=X1.dtype) / B, torch.ones((B,1), device=device, dtype=X2.dtype) / B

        if ot_type == 'exact':
            loss = 0
            for d in range(D):
                X1_d = X1[..., d]
                X2_d = X2[..., d]
                loss += empirical_bures_wasserstein_distance(X1_d, X2_d, reg=reg_sk, ws=a, wt=b, var_weight=var_weight, mean_weight=mean_weight, eps=eps)
            loss = loss / D
            loss = loss.mean()
        elif ot_type == 'upper_bound':
            X1 = X1.permute(2, 0, 1)  # [D, B, T]
            X2 = X2.permute(2, 0, 1)  # [D, B, T]
            a = a.unsqueeze(0).expand(D, -1, -1)  # (D, B, 1)
            b = b.unsqueeze(0).expand(D, -1, -1)  # (D, B, 1)

            loss = batch_empirical_bures_wasserstein_distance(X1, X2, reg=reg_sk, ws=a, wt=b, var_weight=var_weight, mean_weight=mean_weight, eps=eps)
        return loss

    elif distance == 'mmd_linear_per_dim':
        """Linear kernel MMD - Per dimension"""
        loss = 0
        for d in range(D):
            X1_d = X1[..., d]  # [B, T]
            X2_d = X2[..., d]  # [B, T]
            loss += compute_mmd(X1_d, X2_d, linear_kernel)
        loss = loss / D
        return loss

    elif distance == 'mmd_rbf_per_dim':
        """Gaussian RBF kernel MMD - Per dimension"""
        loss = 0
        sigma = reg_sk
        for d in range(D):
            X1_d = X1[..., d]  # [B, T]
            X2_d = X2[..., d]  # [B, T]
            loss += compute_mmd(X1_d, X2_d, gaussian_kernel, sigma=sigma)
        loss = loss / D
        return loss

    elif distance == 'mmd_poly_per_dim':
        """Polynomial kernel MMD - Per dimension"""
        loss = 0
        degree = 3
        for d in range(D):
            X1_d = X1[..., d]  # [B, T]
            X2_d = X2[..., d]  # [B, T]
            loss += compute_mmd(X1_d, X2_d, polynomial_kernel, degree=degree)
        loss = loss / D
        return loss

    elif distance == 'mmd_multi_per_dim':
        """Multi-kernel MMD - Per dimension"""
        loss = 0
        sigmas = [0.1, 1.0, 10.0]
        for d in range(D):
            X1_d = X1[..., d]  # [B, T]
            X2_d = X2[..., d]  # [B, T]
            loss += multi_kernel_mmd(X1_d, X2_d, sigmas)
        loss = loss / D
        return loss

    elif distance == 'kl_per_dim':
        """KL divergence per dimension using PyTorch official KLDivLoss"""
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = 0
        
        for d in range(D):
            # Get all batches for dimension d: [B, T]
            x1_d = X1[:, :, d]  # [B, T] - predictions for dimension d
            x2_d = X2[:, :, d]  # [B, T] - targets for dimension d
            
            # Convert to log probabilities and probabilities
            log_input = F.log_softmax(x1_d, dim=1)    # [B, T] - log probabilities
            target = F.softmax(x2_d, dim=1)           # [B, T] - probabilities
            
            # Compute KL divergence for this dimension
            kl_d = kl_loss(log_input, target)
            loss += kl_d
        
        loss = loss / D  # Average over dimensions
        return loss

    if normalize == 1:
        M = M / M.max()

    if mask_factor > 0:
        mask = torch.ones_like(M) - torch.eye(B, device=device)
        M = M + mask_factor * mask

    a, b = torch.ones((B,), device=device) / B, torch.ones((B,), device=device) / B

    if ot_type == 'sinkhorn':
        pi = ot.sinkhorn(a, b, M, reg=reg_sk, max_iter=numItermax, tol_rel=stopThr).detach()
    elif ot_type == 'emd':
        pi = ot.emd(a, b, M, numItermax=numItermax).detach()
    elif ot_type == 'uot':
        pi = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg_sk, stopThr=stopThr, numItermax=numItermax, reg_m=reg_m).detach()
    elif ot_type == 'uot_mm':
        pi = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m, c=None, reg=0, div='kl', G0=None, numItermax=numItermax, stopThr=stopThr).detach()

    loss = (pi * M).sum()
    return loss


def batch_empirical_bures_wasserstein_distance(
    xs, xt, reg=1e-6, ws=None, wt=None, bias=True, var_weight=1.0, mean_weight=1.0, eps=1e-9
):
    """
    批量计算 Bures-Wasserstein 距离，支持所有维度同时计算
    输入形状: xs, xt: (D, B, T)
             ws, wt: (D, B, 1)
    """
    D_batch, B, T = xs.shape

    # 计算均值 (D, 1, T)
    if bias:
        mxs = torch.matmul(ws.transpose(1, 2), xs) / torch.sum(ws, dim=1, keepdim=True)
        mxt = torch.matmul(wt.transpose(1, 2), xt) / torch.sum(wt, dim=1, keepdim=True)
        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = torch.zeros((D_batch, 1, T), dtype=xs.dtype, device=xs.device)
        mxt = torch.zeros((D_batch, 1, T), dtype=xt.dtype, device=xt.device)

    # 批量协方差计算 (D, T, T)
    Cs = torch.matmul((xs * ws).transpose(1, 2), xs) / torch.sum(ws, dim=1, keepdim=True).transpose(1, 2)
    Cs += reg * torch.eye(T, dtype=xs.dtype, device=xs.device).unsqueeze(0)

    Ct = torch.matmul((xt * wt).transpose(1, 2), xt) / torch.sum(wt, dim=1, keepdim=True).transpose(1, 2)
    Ct += reg * torch.eye(T, dtype=xt.dtype, device=xt.device).unsqueeze(0)

    # 批量距离计算
    W_batch = batch_upper_bound_distance(mxs, mxt, Cs, Ct, var_weight=var_weight, mean_weight=mean_weight, eps=eps)
    return torch.mean(W_batch)  # 对所有维度求均值


def batch_upper_bound_distance(ms, mt, Cs, Ct, var_weight=1.0, mean_weight=1.0, eps=1e-6):
    """
    批量计算 Bures-Wasserstein 距离的上界
    输入形状: ms, mt: (D, 1, T)
             Cs, Ct: (D, T, T)
    输出形状: (D,)
    """
    # 计算均值差平方范数 (D,)
    norm_diff_sq = torch.sum((ms - mt) ** 2, dim=(-1, -2))

    # 计算 B 项 (D,)
    trace_Cs = torch.diagonal(Cs, dim1=-2, dim2=-1).sum(dim=-1)
    trace_Ct = torch.diagonal(Ct, dim1=-2, dim2=-1).sum(dim=-1)
    trace_CsCt = torch.diagonal(torch.matmul(Cs, Ct), dim1=-2, dim2=-1).sum(dim=-1)

    B = trace_Cs + trace_Ct - 2 * torch.sqrt(torch.clip(trace_CsCt, min=eps))
    W = torch.sqrt(torch.clip(mean_weight * norm_diff_sq + var_weight * B, min=eps))

    return W


def empirical_bures_wasserstein_distance(
    xs, xt, reg=1e-6, ws=None, wt=None, bias=True, var_weight=1.0, mean_weight=1.0, eps=1e-9
):
    r"""copy from pot library"""
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    d = xs.shape[1]

    if ws is None:
        ws = nx.ones((xs.shape[0], 1), type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones((xt.shape[0], 1), type_as=xt) / xt.shape[0]

    if bias:
        mxs = nx.dot(ws.T, xs) / nx.sum(ws)
        mxt = nx.dot(wt.T, xt) / nx.sum(wt)

        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = nx.zeros((1, d), type_as=xs)
        mxt = nx.zeros((1, d), type_as=xs)

    Cs = nx.dot((xs * ws).T, xs) / nx.sum(ws) + reg * nx.eye(d, type_as=xs)
    Ct = nx.dot((xt * wt).T, xt) / nx.sum(wt) + reg * nx.eye(d, type_as=xt)

    W = bures_wasserstein_distance(mxs, mxt, Cs, Ct, eps=eps, var_weight=var_weight, mean_weight=mean_weight)
    return W


def cal_distance(ms, mt, Cs, Ct, nx, sqrtm_func, eps=1e-6, var_weight=1.0, mean_weight=1.0):
    Cs12 = sqrtm_func(Cs, eps=eps)
    B = nx.trace(Cs + Ct - 2 * sqrtm_func(dots(Cs12, Ct, Cs12), eps=eps))
    W = nx.sqrt(nx.maximum(mean_weight * nx.norm(ms - mt) ** 2 + var_weight * B, 0))
    return W


def bures_wasserstein_distance(ms, mt, Cs, Ct, eps=1e-6, var_weight=1.0, mean_weight=1.0):
    r"""copy from pot library"""
    ms, mt, Cs, Ct = list_to_array(ms, mt, Cs, Ct)
    nx = get_backend(ms, mt, Cs, Ct)

    try:
        return cal_distance(ms, mt, Cs, Ct, nx, sqrtm_svd_stable, eps=eps, var_weight=var_weight, mean_weight=mean_weight)
    except Exception:
        try:
            return cal_distance(ms, mt, Cs, Ct, nx, sqrtm, eps=eps, var_weight=var_weight, mean_weight=mean_weight)
        except Exception:
            return cal_distance(ms, mt, Cs, Ct, nx, sqrtm_newton_schulz_stable, eps=eps, var_weight=var_weight, mean_weight=mean_weight)


def sqrtm(a, *args, **kwargs):
    L, V = torch.linalg.eigh(a)
    L = torch.sqrt(L)
    # Q[...] = V[...] @ diag(L[...])
    Q = torch.einsum("...jk,...k->...jk", V, L)
    # R[...] = Q[...] @ V[...].T
    return torch.einsum("...jk,...kl->...jl", Q, torch.transpose(V, -1, -2))


def sqrtm_svd(a, eps=1e-9, *args, **kwargs):
    U, S, Vh = torch.linalg.svd(a)
    S = torch.clamp(S, min=eps)
    S_root = torch.sqrt(S)
    return (U * S_root) @ U.t()


def sqrtm_svd_stable(A: torch.Tensor, eps: float = 1e-9, dtype: torch.dtype = torch.float64, *args, **kwargs):
    """稳定SVD法（对称化+双精度+奇异值截断）"""
    A = (A + A.transpose(-1, -2)) * 0.5  # 对称化
    A64 = A.to(dtype)  # 转double提升稳定性
    U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
    S = torch.clamp(S, min=eps)
    S_root = torch.sqrt(S)
    sqrtA64 = (U * S_root[..., None, :]) @ U.transpose(-1, -2)
    return sqrtA64.to(A.dtype)  # 回转原始dtype


def sqrtm_newton_schulz(A: torch.Tensor, num_iters: int = 20, eps: float = 1e-12, *args, **kwargs):
    """牛顿-舒尔茨迭代法"""
    A = (A + A.transpose(-1, -2)) * 0.5 + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)  # 对称化+正则化
    norm = torch.norm(A, p='fro')
    Y = A / norm  # 归一化（谱范数≈1）
    Z = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    
    for _ in range(num_iters):
        T = 0.5 * (3.0 * Z - Y @ Z @ Y)
        Z = T
        Y = 0.5 * (3.0 * Y - Y @ Y @ Y)

    return torch.sqrt(norm) * Y  # 恢复尺度


def sqrtm_newton_schulz_stable(A: torch.Tensor, num_iters: int = 20, eps: float = 1e-12, *args, **kwargs) -> torch.Tensor:
    """
    改进的Newton‑Schulz迭代：先缩放矩阵使其谱范数≈1，并添加正则化
    """
    # 1. 对称化（消除非对称性）
    A = (A + A.transpose(-1, -2)) * 0.5
    d = A.shape[-1]
    
    # 2. 正定化：添加微小正则项（避免奇异矩阵）
    jitter = eps * torch.eye(d, dtype=A.dtype, device=A.device)
    A = A + jitter
    
    # 3. 缩放：使谱范数接近1（通过Frobenius范数缩放）
    norm = torch.norm(A, p='fro')
    if norm > 0:
        A = A / norm  # 现在A的谱范数≤1（正定矩阵的谱范数≤Frobenius范数）
    else:
        return torch.eye(d, dtype=A.dtype, device=A.device)
    
    # 4. 迭代计算（Newton-Schulz）
    Y = A
    Z = torch.eye(d, dtype=A.dtype, device=A.device)
    for _ in range(num_iters):
        T = 0.5 * (3.0 * Z - Y @ Z @ Y)
        Z = T
        Y = 0.5 * (3.0 * Y - Y @ Y @ Y)
    
    # 5. 恢复尺度（sqrt(A) = sqrt(norm) * Y）
    sqrtA = torch.sqrt(norm) * Y
    return sqrtA


@dataclass
class LowRankCov:
    V: torch.Tensor        # (T, r)
    sigma: torch.Tensor    # (r,)
    reg: float
    sqrt_reg: float
    a: torch.Tensor        # (r,) = sqrt(sigma^2 + reg) - sqrt_reg
    mean: torch.Tensor     # (T,)
    trace: torch.Tensor    # Tr(C) = reg*T + sum(sigma^2)


def _weighted_center(X: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    wn = w / w.sum()
    mean = (wn[:, None] * X).sum(0)
    return X - mean, mean


def lowrank_cov_from_samples(
    X: torch.Tensor,
    w: Optional[torch.Tensor],
    reg: float = 1e-6,
    center: bool = True,
    eig_tol_ratio: float = 1e-14
) -> LowRankCov:
    B, T = X.shape
    device, dtype = X.device, X.dtype
    if w is None:
        w = torch.ones(B, device=device, dtype=dtype)
    else:
        w = w.to(device=device, dtype=dtype)
    w = torch.clamp(w, min=0)
    if w.sum() <= 0:
        raise ValueError("Weights sum <= 0")
    if center:
        Xc, mean = _weighted_center(X, w)
    else:
        Xc = X
        mean = torch.zeros(T, device=device, dtype=dtype)

    wn = w / w.sum()
    Y = Xc * torch.sqrt(wn)  # (B,T)

    # Gram
    G = Y @ Y.T
    G = 0.5 * (G + G.T)

    evals, U = torch.linalg.eigh(G)
    evals = torch.clamp(evals, min=0)
    lam_max = evals.max()
    if lam_max == 0:
        return LowRankCov(
            V=torch.zeros(T, 0, device=device, dtype=dtype),
            sigma=torch.zeros(0, device=device, dtype=dtype),
            reg=reg,
            sqrt_reg=reg**0.5,
            a=torch.zeros(0, device=device, dtype=dtype),
            mean=mean,
            trace=torch.tensor(reg*T, device=device, dtype=dtype)
        )
    tol = lam_max * eig_tol_ratio
    keep = evals > tol
    if keep.sum() == 0:
        return LowRankCov(
            V=torch.zeros(T, 0, device=device, dtype=dtype),
            sigma=torch.zeros(0, device=device, dtype=dtype),
            reg=reg,
            sqrt_reg=reg**0.5,
            a=torch.zeros(0, device=device, dtype=dtype),
            mean=mean,
            trace=torch.tensor(reg*T, device=device, dtype=dtype)
        )
    lam = evals[keep]
    U_r = U[:, keep]
    sigma = torch.sqrt(lam)
    # V 正交（理论上），数值上可选再正交
    V = (Y.T @ U_r) / sigma
    # 轻微再正交（提升精度）
    V, _ = torch.linalg.qr(V)  # 注意：再正交后 sigma 不再直接对应对角，需要重估投影谱
    # 重新估 sigma：在新 V 上投影 Cs - reg I
    # Cs - reg I = Y^T Y = (V S)(V S)^T, 我们可以再计算 SVD of (Y @ V)
    YV = Y @ V            # (B, r_new)
    # 现在 YV YV^T 的特征值 = diag(sigma_new^2)
    # 直接对 (YV^T YV) 做特征分解即可
    M_small = YV.T @ YV
    M_small = 0.5 * (M_small + M_small.T)
    s_eigs, U_small = torch.linalg.eigh(M_small)
    s_eigs = torch.clamp(s_eigs, min=0)
    sigma_new = torch.sqrt(s_eigs)
    # 更新 V 到 “更正交+对角表示” 基
    V = V @ U_small
    sigma = sigma_new

    sqrt_reg = reg**0.5
    a = torch.sqrt(sigma**2 + reg) - sqrt_reg
    trace_C = torch.tensor(reg * T, device=device, dtype=dtype) + (sigma**2).sum()
    return LowRankCov(
        V=V, sigma=sigma, reg=reg, sqrt_reg=sqrt_reg, a=a,
        mean=mean, trace=trace_C
    )


def dual_bures_wasserstein_distance(
    Xs: torch.Tensor,
    Xt: torch.Tensor,
    ws: Optional[torch.Tensor] = None,
    wt: Optional[torch.Tensor] = None,
    reg: float = 1e-6,
    eps: float = 1e-9,
    bias: bool = True,
    var_weight: float = 1.0,
    eig_tol_ratio: float = 1e-14,
    snap_rel: float = 1e-10,
    snap_abs: float = 1e-22
) -> torch.Tensor:
    """
    代数子空间法：避免 T×k 乘法带来的误差，在 B<<T 场景下提升精度。
    """
    cov_s = lowrank_cov_from_samples(Xs, ws, reg=reg, center=bias, eig_tol_ratio=eig_tol_ratio)
    cov_t = lowrank_cov_from_samples(Xt, wt, reg=reg, center=bias, eig_tol_ratio=eig_tol_ratio)
    T = Xs.shape[1]
    mean_diff_sq = ((cov_s.mean - cov_t.mean)**2).sum()

    if cov_s.V.numel()==0 and cov_t.V.numel()==0:
        return torch.sqrt(torch.clamp(mean_diff_sq, min=0))

    Vs, Vt = cov_s.V, cov_t.V
    rs, rt = Vs.shape[1], Vt.shape[1]

    # 构造 Vt 在 Vs 正交补上的部分
    if rt == 0:
        # 只有 Vs 低秩
        # 这时 span = Vs，Ct = reg I，K_sub = Cs^{1/2} * reg I * Cs^{1/2} restricted
        # 直接走统一分支也可以
        pass

    if rt > 0 and rs > 0:
        # 去掉 Vs 分量
        proj = Vs @ (Vs.T @ Vt)
        R_raw = Vt - proj
    else:
        R_raw = Vt.clone()

    if R_raw.numel() > 0:
        # 对 R_raw 做 QR
        W, R_up = torch.linalg.qr(R_raw, mode='reduced')  # Vt_perp = W
        r_t_perp = W.shape[1]
    else:
        W = torch.zeros_like(Vt)[:,:0]
        R_up = torch.zeros(0,0, device=Vt.device, dtype=Vt.dtype)
        r_t_perp = 0

    # Vt = Vs C + W R_up
    if rt > 0 and rs > 0:
        C = Vs.T @ Vt  # (rs, rt)
    else:
        C = torch.zeros(rs, rt, device=Vs.device, dtype=Vs.dtype)

    # 在基 B = [Vs, W] 上的维度 k
    k = rs + r_t_perp

    # 构造 Ct_sub
    regI_k = reg * torch.eye(k, device=Vs.device, dtype=Vs.dtype)

    # D_t = diag(t_t^2)
    D_t = (cov_t.sigma**2)  # (rt,)

    # 拆出 Vt 的两个块: Vs 部分  C,  W 部分 R_up
    # Block 形式:
    # [Vs^T; W^T] Vt = [ C ; R_up ]
    # 因此 Ct_sub = reg I_k + [C;R_up] D_t [C;R_up]^T
    if rt > 0:
        CD = C * D_t  if C.numel()>0 else C
        RD = R_up * D_t if R_up.numel()>0 else R_up
        # 上左
        TL = CD @ C.T if rt>0 and rs>0 else torch.zeros(rs, rs, device=Vs.device, dtype=Vs.dtype)
        # 上右
        TR = CD @ R_up.T if (rs>0 and r_t_perp>0) else torch.zeros(rs, r_t_perp, device=Vs.device, dtype=Vs.dtype)
        # 下左
        BL = TR.T
        # 下右
        BR = RD @ R_up.T if r_t_perp>0 else torch.zeros(r_t_perp, r_t_perp, device=Vs.device, dtype=Vs.dtype)
        Ct_sub = regI_k.clone()
        if rs>0:
            Ct_sub[:rs,:rs] += TL
        if rs>0 and r_t_perp>0:
            Ct_sub[:rs, rs:] += TR
            Ct_sub[rs:, :rs] += BL
        if r_t_perp>0:
            Ct_sub[rs:, rs:] += BR
    else:
        Ct_sub = regI_k.clone()

    # Cs^{1/2} 在基 B 中： sqrt_reg I_k + diag(d_s, 0)
    # d_s = sqrt(s_s^2 + reg) - sqrt_reg
    if rs>0:
        Cs_half_sub = cov_s.sqrt_reg * torch.eye(k, device=Vs.device, dtype=Vs.dtype)
        Cs_half_sub[:rs,:rs] += torch.diag(cov_s.a)  # a = sqrt(s^2+reg) - sqrt_reg
    else:
        Cs_half_sub = cov_s.sqrt_reg * torch.eye(k, device=Vs.device, dtype=Vs.dtype)

    # K_sub = Cs_half_sub @ Ct_sub @ Cs_half_sub
    K_sub = Cs_half_sub @ (Ct_sub @ Cs_half_sub)
    K_sub = 0.5 * (K_sub + K_sub.T)

    evals, _ = torch.linalg.eigh(K_sub)
    evals = torch.clamp(evals, min=0)

    # 对齐接近 reg^2 的特征值 (避免 sqrt 放大误差)
    reg_sq = reg * reg
    if reg_sq > 0:
        diff = (evals - reg_sq).abs()
        mask = (diff <= snap_rel * reg_sq) | (diff <= snap_abs)
        evals = torch.where(mask, torch.full_like(evals, reg_sq), evals)

    trace_small = torch.sqrt(evals).sum()
    trace_cross = trace_small + (T - k) * reg  # 补空间 reg

    B_term = cov_s.trace + cov_t.trace - 2.0 * trace_cross
    B_term = torch.clamp(B_term, min=0)
    dist_sq = mean_diff_sq + var_weight * B_term
    return torch.sqrt(torch.clamp(dist_sq, min=0))
