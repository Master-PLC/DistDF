import ot
import torch
from multiprocessing import Pool
from functools import partial


def prob_norm(x, eps=1e-4):
    x = torch.clamp(x, min=0.)
    bias = 0 if x.min() > 0 else eps - x.min()
    x = x + bias
    x = x / x.sum()
    return x


def emd_1d(preds, targets, index, invert=False, dist_scale=0.1, eps=1e-4, device='cpu'):
    N = len(index)
    if N == 0:
        return 0.
    
    pred = preds[index] if not invert else -preds[index]
    target = targets[index] if not invert else -targets[index]
    pred, target = prob_norm(pred, eps), prob_norm(target, eps)

    # time = 1.5
    # seq = torch.arange(N).reshape(-1, 1).float().to(device).requires_grad_(True)
    # M = ot.dist(seq, seq, 'euclidean')
    # M = M / M.max() * dist_scale
    # emd = ot.emd2(pred, target, M)

    # time = 1.46
    # seq = torch.arange(N).reshape(-1, 1).float().to(device).requires_grad_(True)
    # M = ot.dist(seq, seq, 'euclidean')
    # M = M / M.max() * dist_scale
    # G0 = ot.emd_1d(seq, seq, pred, target, metric='euclidean')
    # emd = (G0 * M).sum()

    # time = 0.7
    seq = torch.arange(N).reshape(-1, 1).float().to(device).requires_grad_(True)
    emd = ot.wasserstein_1d(seq, seq, pred, target, p=1, require_sort=False)[0]
    emd = emd / (N - 1) * dist_scale
    return emd


def emd_loss_1d(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.permute(0, 2, 1).reshape(B * D, T)
    target = target.permute(0, 2, 1).reshape(B * D, T)

    emd_losses = []
    for i in range(B * D):
        p = pred[i]
        t = target[i]

        posi = torch.where(t >= 0)[0]
        negi = torch.where(t < 0)[0]

        emd_p = emd_1d(p, t, posi, dist_scale=dist_scale, eps=eps, device=device)
        emd_n = emd_1d(p, t, negi, invert=True, dist_scale=dist_scale, eps=eps, device=device)
        emd_losses.append(emd_p + emd_n)

    return torch.stack(emd_losses)


def emd_1d_batched_align_t(preds, targets, idxi, idxj, invert=False, dist_scale=0.1, eps=1e-4, device='cpu'):
    S, T = targets.shape

    # retrieve part of the target and set the rest to 0
    targets_ = torch.zeros_like(targets)
    preds_ = torch.zeros_like(preds)

    t = targets[idxi, idxj]
    targets_[idxi, idxj] = t if not invert else -t

    p = preds[idxi, idxj]
    preds_[idxi, idxj] = p if not invert else -p

    # ensure positivity and normalize
    preds_ = prob_norm_batched(preds_, eps)
    targets_ = prob_norm_batched(targets_, eps)

    # time = 0.7
    seq = torch.arange(T).reshape(-1, 1).float().repeat(1, S).to(device).requires_grad_(True)  # shape [T, S]
    emd = ot.wasserstein_1d(seq, seq, preds_.T, targets_.T, p=1, require_sort=False)
    emd = emd / (T - 1) * dist_scale
    return emd


def prob_norm_batched(x, eps=1e-4):
    xmin = torch.min(x, dim=1, keepdim=True)[0]
    bias = torch.zeros_like(xmin)
    bias[xmin <= 0] = eps - xmin[xmin <= 0]
    x = x + bias
    x = x / x.sum(dim=1, keepdim=True)
    return x


def emd_loss_1d_batched_align_t(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.permute(0, 2, 1).reshape(B * D, T)
    target = target.permute(0, 2, 1).reshape(B * D, T)

    # align positive part of target
    posi, posj = torch.where(target >= 0)
    emd_p = emd_1d_batched_align_t(pred, target, posi, posj, dist_scale=dist_scale, eps=eps, device=device) if len(posi) > 0 else 0.

    negi, negj = torch.where(target < 0)
    emd_n = emd_1d_batched_align_t(pred, target, negi, negj, invert=True, dist_scale=dist_scale, eps=eps, device=device) if len(negi) > 0 else 0.

    return emd_p + emd_n


def emd_1d_batched_align_h(preds, targets, dist_scale=0.1, eps=1e-4, device='cpu'):
    S, T = targets.shape
    preds = prob_norm_batched(preds, eps)
    targets = prob_norm_batched(targets, eps)

    seq = torch.arange(T).reshape(-1, 1).float().repeat(1, S).to(device).requires_grad_(True)  # shape [T, S]
    emd = ot.wasserstein_1d(seq, seq, preds.T, targets.T, p=1, require_sort=False)
    emd = emd / (T - 1) * dist_scale
    return emd


def emd_loss_1d_batched_align_h(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.permute(0, 2, 1).reshape(B * D, T)
    target = target.permute(0, 2, 1).reshape(B * D, T)

    tp = torch.clamp(target, min=0.)
    pp = torch.clamp(pred, min=0.)
    emd_p = emd_1d_batched_align_h(tp, pp, dist_scale=dist_scale, eps=eps, device=device)

    tn = torch.clamp(-target, min=0.)
    pn = torch.clamp(-pred, min=0.)
    emd_n = emd_1d_batched_align_h(tn, pn, dist_scale=dist_scale, eps=eps, device=device)

    return emd_p + emd_n


def emd_loss_1d_batched_align_all(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.permute(0, 2, 1).reshape(B * D, T)
    target = target.permute(0, 2, 1).reshape(B * D, T)

    target = prob_norm_batched(target, eps)
    pred = prob_norm_batched(pred, eps)

    seq = torch.arange(T).reshape(-1, 1).float().repeat(1, B * D).to(device).requires_grad_(True)  # shape [T, S]
    emd = ot.wasserstein_1d(seq, seq, pred.T, target.T, p=1, require_sort=False)
    emd = emd / (T - 1) * dist_scale
    return emd


def emd_loss_2d_batched_align_t(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.reshape(B, T * D)
    target = target.reshape(B, T * D)

    posi, posj = torch.where(target >= 0)
    emd_p = emd_1d_batched_align_t(pred, target, posi, posj, dist_scale=dist_scale, eps=eps, device=device) if len(posi) > 0 else 0.

    negi, negj = torch.where(target < 0)
    emd_n = emd_1d_batched_align_t(pred, target, negi, negj, invert=True, dist_scale=dist_scale, eps=eps, device=device) if len(negi) > 0 else 0.

    return emd_p + emd_n


def emd_loss_2d_batched_align_h(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.reshape(B, T * D)
    target = target.reshape(B, T * D)

    tp = torch.clamp(target, min=0.)
    pp = torch.clamp(pred, min=0.)
    emd_p = emd_1d_batched_align_h(tp, pp, dist_scale=dist_scale, eps=eps, device=device)

    tn = torch.clamp(-target, min=0.)
    pn = torch.clamp(-pred, min=0.)
    emd_n = emd_1d_batched_align_h(tn, pn, dist_scale=dist_scale, eps=eps, device=device)

    return emd_p + emd_n


def emd_loss_2d_batched_align_all(pred, target, dist_scale=0.1, eps=1e-4, device='cpu'):
    B, T, D = pred.shape
    pred = pred.reshape(B, T * D)
    target = target.reshape(B, T * D)

    target = prob_norm_batched(target, eps)
    pred = prob_norm_batched(pred, eps)
    
    seq = torch.arange(T * D).reshape(-1, 1).float().repeat(1, B).to(device).requires_grad_(True)
    emd = ot.wasserstein_1d(seq, seq, pred.T, target.T, p=1, require_sort=False)
    emd = emd / (T * D - 1) * dist_scale
    return emd
