import os
import time
import warnings

import numpy as np
import yaml
from copy import deepcopy
import torch
import torch.nn as nn
import torch.profiler as profiler
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import MODEL_REQUIRES_CYCLE
from torch import optim
from utils.dilate_loss import dilate_loss
from utils.dilate_loss_cuda import DilateLossCUDA
# from utils.dilate_loss_cache import dilate_loss
from utils.soft_dtw_cuda import SoftDTW
from utils.dtw_cuda import DTW
from utils.dpp_loss import dpp_loss
from utils.fft_ot import cal_wasserstein
from utils.fourier_koopman import fourier_loss
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.ot_dist import *
from utils.polynomial import chebyshev_torch, hermite_torch, laguerre_torch, leg_torch, pca_torch, Basis_Cache, ica_torch, robust_ica_torch, robust_pca_torch, svd_torch, random_torch, Random_Cache, fa_torch
from utils.tools import EarlyStopping, visual, Scheduler, adjust_learning_rate

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        # total_loss = np.average(total_loss)
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

    def initialize_cache(self, train_data):
        cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'random':
                cache = Random_Cache(
                    rank_ratio=self.args.rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.pred_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
            elif self.args.auxi_type == 'fa':
                cache = Basis_Cache(train_data.fa_components, train_data.initializer, mean=train_data.fa_mean, device=self.device)
            elif self.args.auxi_type == 'pca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
            elif self.args.auxi_type == 'robustpca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, mean=train_data.rpca_mean, device=self.device)
            elif self.args.auxi_type == 'svd':
                cache = Basis_Cache(train_data.svd_components, train_data.initializer, device=self.device)
            elif self.args.auxi_type == 'ica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, mean=train_data.ica_mean, whitening=train_data.whitening, device=self.device)
            elif self.args.auxi_type == 'robustica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, device=self.device)
        return cache

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        cache = self.initialize_cache(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        model_state_last_effective = None
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        if self.args.auxi_mode == 'fourier_koopman':
            freqs = nn.Parameter(torch.tensor(train_data.freqs, device=self.device, dtype=torch.float32))
            model_optim.add_param_group({'params': freqs, 'lr': self.args.learning_rate})
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()
        if self.args.auxi_mode == 'soft_dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "SoftDTW only supports GPU"
            sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        elif self.args.auxi_mode == 'dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DTW only supports GPU"
            dtw = DTW(use_cuda=True, bandwidth=0.1)
        elif self.args.auxi_mode == 'dilate_cuda':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DILATE only supports GPU"
            dilate_cuda = DilateLossCUDA(alpha=self.args.dilate_alpha, gamma=self.args.gamma, bandwidth=0)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            has_nan_in_epoch = False
            train_loss = []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                loss = 0
                if self.args.rec_lambda:
                    if prof is not None:
                        with profiler.record_function("rec_loss_forward_pass"):
                            loss_rec = criterion(outputs, batch_y)
                    else:
                        loss_rec = criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)
                if self.step % self.log_step == 0:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)

                if self.args.l1_weight and attn:
                    loss += self.args.l1_weight * attn[0]

                if self.args.auxi_lambda:
                    if self.args.joint_forecast:  # joint distribution forecasting
                        outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1)  # [B, S+P, D]
                        batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1)  # [B, S+P, D]

                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # shape: [B, P, D]

                    elif self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                        elif self.args.auxi_type == 'complex-phase':
                            loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()  
                        elif self.args.auxi_type == 'complex-mag-phase':
                            loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                            loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])  # shape: [2, B, P//2+1, D]
                        elif self.args.auxi_type == 'phase':
                            loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()  # shape: [B, P//2+1, D]
                        elif self.args.auxi_type == 'mag':
                            loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()  # shape: [B, P//2+1, D]
                        elif self.args.auxi_type == 'mag-phase':
                            loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                            loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])  # shape: [2, B, P//2+1, D]
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "rfft-D":
                        loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)  # shape: [B, P, D//2+1]

                    elif self.args.auxi_mode == "rfft-2D":
                        loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)  # shape: [B, P, D//2+1]

                    elif self.args.auxi_mode == "basis":
                        kwargs = {'degree': self.args.leg_degree, 'device': self.device}
                        if self.args.auxi_type == "legendre":
                            loss_auxi = leg_torch(outputs, **kwargs) - leg_torch(batch_y, **kwargs)  # shape: [B*D, degree+1]
                        elif self.args.auxi_type == "chebyshev":
                            loss_auxi = chebyshev_torch(outputs, **kwargs) - chebyshev_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "hermite":
                            loss_auxi = hermite_torch(outputs, **kwargs) - hermite_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "laguerre":
                            loss_auxi = laguerre_torch(outputs, **kwargs) - laguerre_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "random":
                            kwargs = {'pca_dim': self.args.pca_dim, 'random_cache': cache, 'device': self.device}
                            loss_auxi = random_torch(outputs, **kwargs) - random_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "fa":
                            kwargs = {'pca_dim': self.args.pca_dim, 'fa_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = fa_torch(outputs, **kwargs) - fa_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'use_weights': self.args.use_weights, 
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            if prof is not None:
                                with profiler.record_function("auxi_loss_forward_pass"):
                                    loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                            else:
                                loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "robustpca":
                            kwargs = {'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = robust_pca_torch(outputs, **kwargs) - robust_pca_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "svd":
                            kwargs = {'pca_dim': self.args.pca_dim, 'svd_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = svd_torch(outputs, **kwargs) - svd_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "ica":
                            kwargs = {'pca_dim': self.args.pca_dim, 'ica_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = ica_torch(outputs, **kwargs) - ica_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "robustica":
                            kwargs = {'pca_dim': self.args.pca_dim, 'ica_cache': cache, 'reinit': self.args.reinit, 'device': self.device}
                            loss_auxi = robust_ica_torch(outputs, **kwargs) - robust_ica_torch(batch_y, **kwargs)
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "ot":
                        kwargs = {'dist_scale': self.args.dist_scale, 'device': self.device}
                        if self.args.auxi_type == "emd1d_t":
                            loss_auxi = emd_loss_1d_batched_align_t(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd1d_h":
                            loss_auxi = emd_loss_1d_batched_align_h(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd1d_all":
                            loss_auxi = emd_loss_1d_batched_align_all(outputs, batch_y, **kwargs)

                        elif self.args.auxi_type == "emd2d_h":
                            loss_auxi = emd_loss_2d_batched_align_h(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd2d_t":
                            loss_auxi = emd_loss_2d_batched_align_t(outputs, batch_y, **kwargs)
                        elif self.args.auxi_type == "emd2d_all":
                            loss_auxi = emd_loss_2d_batched_align_all(outputs, batch_y, **kwargs)

                        elif self.args.auxi_type == "emd1d_h_learn_proj":
                            outputs_proj = self.model.project(outputs)
                            batch_y_proj = self.model.project(batch_y)
                            loss_auxi = emd_loss_1d_batched_align_h(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_t_learn_proj":
                            outputs_proj = self.model.project(outputs)
                            batch_y_proj = self.model.project(batch_y)
                            loss_auxi = emd_loss_1d_batched_align_t(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_all_learn_proj":
                            outputs_proj = self.model.project(outputs)
                            batch_y_proj = self.model.project(batch_y)
                            loss_auxi = emd_loss_1d_batched_align_all(outputs_proj, batch_y_proj, **kwargs)

                        elif self.args.auxi_type == "emd1d_h_pca_proj":
                            n_feats, rank_ratio = self.args.c_out, self.args.rank_ratio
                            low_rank = int(n_feats * rank_ratio)
                            outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                            batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                            loss_auxi = emd_loss_1d_batched_align_h(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_t_pca_proj":
                            n_feats, rank_ratio = self.args.c_out, self.args.rank_ratio
                            low_rank = int(n_feats * rank_ratio)
                            outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                            batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                            loss_auxi = emd_loss_1d_batched_align_t(outputs_proj, batch_y_proj, **kwargs)
                        elif self.args.auxi_type == "emd1d_all_pca_proj":
                            n_feats, rank_ratio = self.args.c_out, self.args.rank_ratio
                            low_rank = int(n_feats * rank_ratio)
                            outputs_proj = torch.matmul(outputs, torch.pca_lowrank(outputs.reshape(-1, n_feats), low_rank)[-1])
                            batch_y_proj = torch.matmul(batch_y, torch.pca_lowrank(batch_y.reshape(-1, n_feats), low_rank)[-1])
                            loss_auxi = emd_loss_1d_batched_align_all(outputs_proj, batch_y_proj, **kwargs)

                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "fft_ot":
                        loss_auxi = cal_wasserstein(
                            outputs, batch_y, self.args.distance, ot_type=self.args.ot_type, normalize=self.args.normalize, 
                            mask_factor=self.args.mask_factor, reg_sk=self.args.reg_sk, stopThr=self.args.stopThr, numItermax=self.args.numItermax, 
                            var_weight=self.args.var_weight, mean_weight=self.args.mean_weight
                        )

                    elif self.args.auxi_mode == "fourier_koopman":
                        loss_auxi = fourier_loss(outputs, batch_y, freqs, device=self.device)

                    elif self.args.auxi_mode == "dilate":
                        loss_auxi, _, _ = dilate_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "dpp":
                        loss_auxi = dpp_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "soft_dtw":
                        loss_auxi = sdtw(outputs, batch_y)

                    elif self.args.auxi_mode == "dtw":
                        loss_auxi = dtw(outputs, batch_y)[0].mean()

                    elif self.args.auxi_mode == "dilate_cuda":
                        loss_auxi = dilate_cuda(outputs, batch_y)

                    else:
                        raise NotImplementedError

                    if self.mask is not None:
                        loss_auxi *= self.mask

                    if self.args.auxi_loss == "MAE":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.args.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                    elif self.args.auxi_loss == "None":
                        pass
                    else:
                        raise NotImplementedError

                    loss += self.args.auxi_lambda * loss_auxi
                else:
                    loss_auxi = torch.tensor(1e4)
                if self.step % self.log_step == 0:
                    self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN or Inf, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                train_loss.append(loss.item())
                self.writer.add_scalar(f'{self.pred_len}/train/loss_iter', loss.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss: {:.7f}".format(
                            i + 1, self.epoch, loss_rec.item(), loss_auxi.item(), loss.item()
                        )
                    )
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                if prof is not None:
                    with profiler.record_function("backward_pass"):
                        loss.backward()
                else:
                    loss.backward()
                model_optim.step()

                if prof is not None:
                    prof.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)

            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    self.epoch, self.step, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, prof=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_data.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if prof is not None:
                    prof.step()

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())
            if self.args.auxi_mode == 'basis' and self.args.auxi_type == 'pca':
                train_data, _ = self._get_data(flag='train')
                pca_components = train_data.pca_components
                np.save(os.path.join(res_path, 'pca_components.npy'), pca_components)

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
