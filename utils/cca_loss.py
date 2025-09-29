import torch
import torch.nn.functional as F


def cal_corr(X, Y, rank_ratio=1.0, r1=1e-3, r2=1e-3, eps=1e-9, device='cpu', n_retry=3):
    X, Y = X.t(), Y.t()

    o1 = o2 = X.size(0)
    m = X.size(1)

    Xbar = X - X.mean(dim=1, keepdim=True)
    Ybar = Y - Y.mean(dim=1, keepdim=True)

    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(Xbar, Ybar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(Xbar, Xbar.t()) + r1 * torch.eye(o1, device=device, dtype=X.dtype)
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(Ybar, Ybar.t()) + r2 * torch.eye(o2, device=device, dtype=Y.dtype)

    # Use torch.linalg.eigh to replace deprecated torch.symeig
    for _ in range(n_retry):
        try:
            D1, V1 = torch.linalg.eigh(SigmaHat11)
            D2, V2 = torch.linalg.eigh(SigmaHat22)
            break
        except RuntimeError as e:
            SigmaHat11 = SigmaHat11 + r1 * torch.eye(o1, device=device, dtype=X.dtype)
            SigmaHat22 = SigmaHat22 + r2 * torch.eye(o2, device=device, dtype=Y.dtype)
    else:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # Added to increase stability
    posInd1 = torch.gt(D1, eps).nonzero(as_tuple=True)[0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero(as_tuple=True)[0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
    if rank_ratio == 1.0:
        # tmp = torch.matmul(Tval.t(), Tval)
        # corr = torch.trace(torch.sqrt(tmp))
        tmp = torch.matmul(Tval.t(), Tval)
        eigenvals = torch.linalg.eigvals(tmp)
        corr = torch.sum(torch.sqrt(torch.clamp(eigenvals.real, min=0)))
    else:
        outdim_size = int(rank_ratio * o1)
        trace_TT = torch.matmul(Tval.t(), Tval)
        # trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(device))
        U, V = torch.linalg.eigh(trace_TT)
        U = torch.where(U>eps, U, (torch.ones(U.shape, dtype=torch.double)*eps).to(device))
        U = U.topk(outdim_size).values
        corr = torch.sum(torch.sqrt(U))
    return -corr


def cal_corr_svd(X, Y, rank_ratio=1.0, r1=1e-3, r2=1e-3, eps=1e-9, device='cpu', n_retry=3):
    X, Y = X.t(), Y.t()
    o1 = o2 = X.size(0)
    m = X.size(1)

    Xbar = X - X.mean(dim=1, keepdim=True)
    Ybar = Y - Y.mean(dim=1, keepdim=True)

    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(Xbar, Ybar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(Xbar, Xbar.t()) + r1 * torch.eye(o1, device=device, dtype=X.dtype)
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(Ybar, Ybar.t()) + r2 * torch.eye(o2, device=device, dtype=Y.dtype)

    # SVD-based CCA: stabilize and avoid eigenvalue issues
    for _ in range(n_retry):
        try:
            # Compute matrix square root inverse via SVD
            U1, S1, Vh1 = torch.linalg.svd(SigmaHat11)
            U2, S2, Vh2 = torch.linalg.svd(SigmaHat22)
            # Clamp singular values for numerical stability
            S1_inv_sqrt = torch.diag(torch.clamp(S1, min=eps).pow(-0.5))
            S2_inv_sqrt = torch.diag(torch.clamp(S2, min=eps).pow(-0.5))
            SigmaHat11RootInv = U1 @ S1_inv_sqrt @ U1.t()
            SigmaHat22RootInv = U2 @ S2_inv_sqrt @ U2.t()
            break
        except RuntimeError as e:
            SigmaHat11 = SigmaHat11 + r1 * torch.eye(o1, device=device, dtype=X.dtype)
            SigmaHat22 = SigmaHat22 + r2 * torch.eye(o2, device=device, dtype=Y.dtype)
    else:
        return torch.tensor(0.0, requires_grad=True, device=device)

    Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
    # SVD on Tval to get canonical correlations
    U, S, Vh = torch.linalg.svd(Tval)
    # S contains canonical correlations
    if rank_ratio == 1.0:
        corr = S.sum()
    else:
        outdim_size = int(rank_ratio * o1)
        corr = S[:outdim_size].sum()
    return -corr


def cal_corr_cosine(X, Y, rank_ratio=1.0, r1=1e-3, r2=1e-3, eps=1e-9, device='cpu', n_retry=3):
    corr = F.cosine_similarity(X, Y, dim=-1)
    corr = corr.mean()
    return -corr


def cal_corr_pearson(X, Y, rank_ratio=1.0, r1=1e-3, r2=1e-3, eps=1e-9, device='cpu', n_retry=3):
    X_centered = X - X.mean(dim=1, keepdim=True)
    Y_centered = Y - Y.mean(dim=1, keepdim=True)
    covariance = (X_centered * Y_centered).mean(dim=1)
    X_std = X_centered.std(dim=1)
    Y_std = Y_centered.std(dim=1)
    corr = covariance / (X_std * Y_std + eps)  # [B]
    corr = corr.mean()
    return -corr


def robust_cca_loss(X, Y, rank_ratio=1.0, r1=1e-3, r2=1e-3, eps=1e-9, device='cpu', n_retry=3):
    """
    Computes a robust CCA loss for the case B < D.
    X, Y: [B, D] tensors, already centered.
    k: Number of canonical correlations to sum. If None, uses all B.
    """
    B, D = X.shape
    # assert Y.shape == (B, D), "X and Y must have the same shape"
    # assert B < D, "This implementation is for B < D"

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # 1. QR Decomposition for orthogonal basis
    # Q_x will be a B x B orthogonal matrix (an orthonormal basis for the column space of X^T)
    Q_x, _ = torch.linalg.qr(X.t()) # X.t() is [D, B]
    Q_y, _ = torch.linalg.qr(Y.t()) # Y.t() is [D, B]

    # 2. Compute the correlation between the orthonormal bases
    # Q_x.t() @ Q_y is a B x B matrix
    T = Q_x.t() @ Q_y

    # 3. SVD of the correlation matrix
    # The singular values of T are the canonical correlations
    _, S, _ = torch.linalg.svd(T)

    # 4. Sum the canonical correlations
    if rank_ratio < 1.0:
        k = int(rank_ratio * B)
        corr = torch.sum(S[:k])
    else:
        corr = torch.sum(S)

    return -corr


def cca_loss(X, Y, align_type=1, rank_ratio=1.0, device='cpu', r1=1e-3, r2=1e-3, eps=1e-9, n_retry=3, stride=1, loss_type='cosine'):
    """
    X: [B, S, D], Y: [B, P, D]
    """
    if loss_type == 'svd':
        cal_corr_func = cal_corr_svd
    elif loss_type == 'eigh':
        cal_corr_func = cal_corr
    elif loss_type == 'cosine':
        cal_corr_func = cal_corr_cosine
    elif loss_type == 'pearson':
        cal_corr_func = cal_corr_pearson
    elif loss_type == 'dual':
        cal_corr_func = robust_cca_loss

    if align_type == 0:
        X = X.mean(dim=1)
        Y = Y.mean(dim=1)
    elif align_type == 1:
        X = X[:, -1]
        Y = Y[:, 0]
    elif align_type == 2:
        X = X[:, -1]
        Y = Y[:, -1]
    elif align_type == 3:
        X = X[:, 0]
        Y = Y[:, 0]
    elif align_type == 4:
        X = X.sum(dim=1)
        Y = Y.sum(dim=1)
    elif align_type == 5:
        S, P, D = X.size(1), Y.size(1), X.size(2)
        corr_list = []
        if S <= P:
            # 滑动窗口在Y，窗口长度为S
            for idx in range(0, P - S + 1, stride):
                Y_ = Y[:, idx:idx+S, :]  # [B, S, D]
                X_flat = X.reshape(-1, D)
                Y_flat = Y_.reshape(-1, D)
                corr = cal_corr_func(X_flat, Y_flat, rank_ratio=1.0, r1=r1, r2=r2, eps=eps, device=device, n_retry=n_retry)
                corr_list.append(corr)
        elif S > P:
            # 滑动窗口在X，窗口长度为P
            for idx in range(0, S - P + 1, stride):
                X_ = X[:, idx:idx+P, :]  # [B, P, D]
                X_flat = X_.reshape(-1, D)
                Y_flat = Y.reshape(-1, D)
                corr = cal_corr_func(X_flat, Y_flat, rank_ratio=1.0, r1=r1, r2=r2, eps=eps, device=device, n_retry=n_retry)
                corr_list.append(corr)
        loss = torch.stack(corr_list).mean()
        return loss

    elif align_type == 6:
        X = X[torch.arange(X.size(0)), torch.randint(X.size(1), (X.size(0),))]
        Y = Y[torch.arange(Y.size(0)), torch.randint(Y.size(1), (Y.size(0),))]

    elif align_type == 7:
        pass

    loss = cal_corr_func(X, Y, rank_ratio=1.0, r1=r1, r2=r2, eps=eps, device=device, n_retry=n_retry)
    return loss



def channel_decorrelation_loss(X, eps=1e-8, p=1):
    """
    计算时序数据各通道间相关性的去相关loss。

    Args:
        X: [B, T, D]，B为batch，T为时间步，D为通道数
        eps: 防止除0
        p: 1为L1范数（绝对值），2为L2范数（平方）

    Returns:
        loss: 所有通道两两间相关系数的均值（L1或L2）
    """
    B, T, D = X.shape
    X_flat = X.reshape(-1, D)  # [B*T, D]
    # 中心化
    X_centered = X_flat - X_flat.mean(dim=0, keepdim=True)  # [B*T, D]
    # 标准化
    X_std = X_centered.std(dim=0, keepdim=True) + eps       # [1, D]
    X_norm = X_centered / X_std                             # [B*T, D]
    # 相关系数矩阵 [D, D]
    corr_matrix = (X_norm.T @ X_norm) / (B * T - 1)
    # 忽略对角线，只统计通道间相关性
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=X.device)
    off_diag_corr = corr_matrix[off_diag_mask]  # [D*D-D]
    # L1或L2范数
    if p == 1:
        loss = off_diag_corr.abs().mean()
    elif p == 2:
        loss = off_diag_corr.pow(2).mean()
    else:
        raise ValueError("p must be 1 or 2.")
    return loss