import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)

    # # Naive implementation with O(n) time complexity
    # def forward(self, x):
    #     # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
    #     s = x[:, 0, :]
    #     res = [s.unsqueeze(1)]
    #     for t in range(1, x.shape[1]):
    #         xt = x[:, t, :]
    #         s = self.alpha * xt + (1 - self.alpha) * s
    #         res.append(s.unsqueeze(1))
    #     return torch.cat(res, dim=1)


class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    """
    def __init__(self, alpha, beta):
        super(DEMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        # self.beta = nn.Parameter(beta)      # Learnable beta
        self.alpha = alpha.to(device='cuda')
        self.beta = beta.to(device='cuda')

    def forward(self, x):
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        # self.beta.data.clamp_(0, 1)         # Clamp learnable beta to [0, 1]
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
            b = self.beta * (s - s_prev) + (1 - self.beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)


class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average