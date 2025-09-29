import torch
import torch.nn as nn
from layers.Decomp import DECOMP
from layers.RevIN import RevIN


class NetworkCNN(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, trend):
        super(NetworkCNN, self).__init__()
        
        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num//trend, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num//trend)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num//trend * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        
        x = x.permute(0,2,1) # to [Batch, Channel, Input]
        
        # Channel split
        B = x.shape[0] # Batch size
        C = x.shape[1] # Channel size
        I = x.shape[2] # Input size
        x = torch.reshape(x, (B*C, I)) # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [Batch and Channel, Patch_num, Patch_len]
        
        # Patch Embedding
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        res = x

        # CNN Depthwise
        x = self.conv1(x)
        x = self.gelu2(x)
        x = self.bn2(x)

        # Residual Stream
        res = self.fc2(res)
        x = x + res

        # CNN Pointwise
        x = self.conv2(x)
        x = self.gelu3(x)
        x = self.bn3(x)

        # Flatten Head
        x = self.flatten1(x)
        x = self.fc3(x)
        x = self.gelu4(x)
        x = self.fc4(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x


class NetworkMLP(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(NetworkMLP, self).__init__()
        
        # Parameters
        self.pred_len = pred_len

        # Linear Stream
        # MLP
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc3 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        
        x = x.permute(0,2,1) # to [Batch, Channel, Input]
        
        # Channel split
        B = x.shape[0] # Batch size
        C = x.shape[1] # Channel size
        I = x.shape[2] # Input size
        x = torch.reshape(x, (B*C, I)) # [Batch and Channel, Input]

        # Linear Stream
        # MLP
        x = self.fc1(x)
        x = self.avgpool1(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.avgpool2(x)
        x = self.ln2(x)

        x = self.fc3(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend
        
        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I)) # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]
        
        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream

    def forward(self, x, *args):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x