import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import wandb
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.distribution import Normal, Bernoulli, Logistic256, MixtureLogistic256, DiscretizedGaussian
from model.ddgm import DiffusionPrior


def compute_sample_kl(q, p, z, reduce_dim=True):
    # q is define per point -> returns a scalar
    log_q = q.log_prob(z, reduce_dim=reduce_dim)
    if hasattr(p, 'log_prob'):
        log_p = p.log_prob(z, reduce_dim=reduce_dim)
    else:
        log_p = p.reshape(p.shape[0], -1)
        if reduce_dim:
            log_p = log_p.sum(1)
    if reduce_dim:
        assert len(log_q.shape) == 1, f'Log q should be a vector, got shape {log_q.shape} instead'
        assert len(log_p.shape) == 1, f'Log p should be a vector, got shape {log_p.shape} instead'
    else:
        assert log_q.shape[1] == log_p.shape[1]
    # compute sample kl (averaged over the mini-batch):
    kl = log_q - log_p
    assert kl.shape[0] == z.shape[0]
    return kl


class LADDER_VAE(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 likelihood,
                 beta_start,
                 beta_end,
                 warmup,
                 is_k,
                 latent_scales,
                 **kwargs):
        super(LADDER_VAE, self).__init__()
        self.__dict__.update(locals())
        self.encoder = encoder
        self.decoder = decoder
        self.current_epoch = None

        self.likelihood = {
            'bernoulli': Bernoulli,
            'gaussian': lambda mu, logvar: Normal(mu, torch.log(F.softplus(torch.clamp(logvar, -5., 5), 0.7))),
            'logistic': Logistic256,
            'logistic_mixture': MixtureLogistic256,
            'discretized_gaussian': DiscretizedGaussian,
        }[likelihood]
        self.fid = None

    def encode(self, batch):
        x = batch[0]
        encoder_s = self.encoder(x)
        return encoder_s

    def forward(self, batch, mode='train'):
        encoder_s = self.encode(batch)
        p_xz_params, p_dist, q_dist, z_samples = self.decoder(encoder_s, mode = mode)
        p_xz = self.likelihood(*p_xz_params)
        x_sample = self.get_x_from_px(p_xz)
        return x_sample, p_xz, z_samples, q_dist, p_dist

    def generate_x(self, N=25, t=None):
        enc_s = [None for _ in range(sum(self.latent_scales))]
        p_xz_params, _, _, _ = self.decoder(enc_s, mode='sample', N=N, t=t)
        p_xz = self.likelihood(*p_xz_params)
        return self.get_x_from_px(p_xz)

    def get_x_from_px(self, p_xz):
        # for binary output we get the mean and scale to [-1, 1]
        if isinstance(p_xz, Bernoulli):
            x_rec = p_xz.get_E()
            x_rec = x_rec * 2 - 1
        # for the rest sample from the likelihood
        else:
            x_rec = p_xz.sample()
        return x_rec

    def eval_log_p_x(self, p_xz, x):
        # for bernoulli we need to scale to [0, 1] and get a sample.
        if isinstance(p_xz, Bernoulli):
            x = 0.5 * (x + 1)
            x = torch.bernoulli(x)
        return p_xz.log_prob(x)

    def kl(self, fwd_output, reduce_dim=True):
        x_sample, _, z_sample, q_dist, p_dist = fwd_output
        logs = {}
        kl_vals = []
        l = 0
        for z, q, p in zip(z_sample, q_dist, p_dist):
            if hasattr(q, 'kl'):
                try:
                    MB = z.shape[0]
                    kl = q.kl(p).reshape(MB, -1)  # MB x dim
                    if reduce_dim:
                        kl = kl.sum(1)
                except:
                    kl = compute_sample_kl(q, p, z, reduce_dim=reduce_dim)
            else:
                kl = compute_sample_kl(q, p, z, reduce_dim=reduce_dim)
            kl_vals.append(kl)

            bpd_coef = 1. / np.log(2.) / np.prod(x_sample.shape[1:])
            n = '-' + str(l) if l > 0 else ''
            logs[f'kl_L{n}'] = kl.mean() * bpd_coef
            l += 1
        # return tensor MB x L and the logs
        # ddgm loss
        if hasattr(p_dist[0], 'log_prob'):
            nll_ddgm = - p_dist[0].log_prob(z_sample[0])
        else:
            nll_ddgm = - p_dist[0]
        logs['nll_z_L_prior'] = nll_ddgm.reshape(nll_ddgm.shape[0], -1).sum(1).mean(0).detach()
        if reduce_dim:
            kl_vals = torch.stack(kl_vals, 1)
        return kl_vals, logs

    def calculate_active_units(self, fwd_output):
        _, _, z_sample, q_dist, p_dist = fwd_output
        MB = z_sample[0].shape[0]
        logs = {}
        # AU as in Burda'15 (Cov_x E[z])
        E_q = [q.get_E().reshape(MB, -1).data.cpu() for q in q_dist]
        unit_cnt = sum([mu.shape[1] for mu in E_q])
        cov = [torch.var(mu, dim=0).float() for mu in E_q]
        logs['misc/% Active units (Cov > 0.01)'] = sum(torch.cat(cov) > 0.01)/unit_cnt
        logs['misc/% Active units (Cov > 0.001)'] = sum(torch.cat(cov) > 0.001)/unit_cnt
        return logs
    
    def calculate_collapsed_dim(self, fwd_output):
        _, _, z_sample, q_dist, p_dist = fwd_output
        MB = z_sample[0].shape[0]
        logs = {}
        vec_kl_tot = []
        for z, q_curr, p_curr in zip(z_sample, q_dist, p_dist):
            vec_kl_tot.append( compute_sample_kl(q_curr, p_curr, z, reduce_dim=False))
        tot_kl = torch.cat(vec_kl_tot, 1)
        num_latents = tot_kl.shape[1]
        eps_grid = torch.linspace(1e-5, 1.5, 50)   
        res = {
            d:[] for d in [0.05, 0.01]
        }
        for delta in [0.05, 0.01]:
            for eps in eps_grid:
                num_small = (tot_kl < eps).sum(0)
                collapsed = (num_small >= (1-delta) * tot_kl.shape[0]).sum().item()
                res[delta].append(collapsed/num_latents)
            
        for delta in [0.05, 0.01]:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
            ax.plot(eps_grid, res[delta])
            ax.grid(True)
            logs[f'Collapse/% collapsed dim (delta ={delta:.2f})']  = wandb.Image(fig)
            plt.close()
            
        # save raw data as file
        torch.save(res, os.path.join(wandb.run.dir, 'collapsed_dims.pt'))
        wandb.save(os.path.join(wandb.run.dir, 'collapsed_dims.pt'))
        return logs

    def compute_loss(self, batch, fwd_output, beta):
        x_sample = fwd_output[0]
        p_xz = fwd_output[1]
        bpd_coef = 1. / np.log(2.) / np.prod(x_sample.shape[1:])
        logs = {}
        # data term
        re = - self.eval_log_p_x(p_xz, batch[0]).mean(0)  # MB -> 1
        # KL-divergence (L values)
        kl, logs_kl = self.kl(fwd_output)
        kl = kl.sum(1).mean()  # Mb x L -> 1
        logs.update(logs_kl)

        loss = re + beta * kl
        loss_per_pixel = loss * bpd_coef
        loss_logs = {
            'reconstruction': re.data,
            'kl': kl.sum().data,
            'loss': loss.data,
            'loss_per_pixel': loss_per_pixel.data,
        }
        logs.update(loss_logs)
        return loss_per_pixel, logs

    def get_beta(self):
        beta = self.beta_end
        if self.current_epoch is not None:
            if self.current_epoch < self.warmup:
                dlt = (self.beta_end - self.beta_start) / self.warmup
                beta = self.beta_start + self.current_epoch * dlt
        return beta

    def train_step(self, batch, mode='train'):
        MB = batch[0].shape[0]
        fwd_output = self.forward(batch, mode=mode)
        x_sample, p_xz, z_samples, q_dist, p_dist = fwd_output
        beta = 1
        if mode == 'train':
            beta = self.get_beta()
        loss, logs = self.compute_loss(batch, fwd_output, beta)
        if mode == 'val':
            logs.update(self.calculate_active_units(fwd_output))
            for l in range(1):
                n = '-' + str(l) if l > 0 else ''
                if hasattr(p_dist[l], 'log_var'):
                    lv = p_dist[l].log_var
                    lv = lv.reshape(lv.shape[0], -1).mean(0).data.detach().cpu()
                    logs[f'misc/lnVar_pz_L{n}_pp'] = lv.mean()
                if hasattr(q_dist[l], 'log_var'):
                    lv = q_dist[l].log_var.reshape(MB, -1).mean(0).data.detach().cpu()
                    logs[f'misc/lnVar_qz_L{n}_pp'] = lv.mean()
        return loss, logs

    def test_step(self, batch, compute_fid=True):
        """
        Here we return all metrics sumed over the batch
        :param batch:
        :return: logs (dict)
        """
        # Forward
        fwd_output = self.forward(batch, mode='test')
        x_rec, p_xz = fwd_output[0], fwd_output[1]
        x = batch[0]
        MB = x.shape[0]

        # Reconstructions and KL term
        kl, _ = self.kl(fwd_output)
        logs = {
            'elbo': - self.eval_log_p_x(p_xz, x).sum(0) + kl.sum(),
        }

        # Compute IWAE
        nll, nll_logs = self.estimate_nll(batch, self.is_k)
        logs['nll'] = nll.sum(0)
        logs.update(nll_logs)

        if compute_fid:
            # Compute FID
            samples = self.generate_x(MB).data
            # rescale to [0, 255]
            samples = (255 * 0.5 * (samples + 1)).type(torch.uint8)
            true_data = (255 * 0.5 * (x + 1)).type(torch.uint8)
            if self.fid is None:
                self.fid = torchmetrics.image.fid.FrechetInceptionDistance()
                self.fid = self.fid.to(x.device)

            if samples.shape[1] == 1:
                samples = samples.repeat(1, 3, 1, 1)
                true_data = true_data.repeat(1, 3, 1, 1)
            self.fid.update(true_data, real=True)
            self.fid.update(samples, real=False)
        return logs

    def eval_fid_on_dset(self, loader, device, temp=1.0):
        fid = torchmetrics.image.fid.FrechetInceptionDistance()
        fid = fid.to(device)
        for batch in tqdm(loader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            x = batch[0]
            MB = x.shape[0]
            samples = self.generate_x(MB, t=temp).data
            # rescale to [0, 255]
            samples = (255 * 0.5 * (samples + 1)).type(torch.uint8)
            true_data = (255 * 0.5 * (x + 1)).type(torch.uint8)
            if samples.shape[1] == 1:
                samples = samples.repeat(1, 3, 1, 1)
                true_data = true_data.repeat(1, 3, 1, 1)
            fid.update(true_data, real=True)
            fid.update(samples, real=False)
        final_fid = fid.compute()
        return final_fid
            
    def estimate_nll(self, batch, K=100):
        """
        Estimate NLL by importance sampling
        :param X: mini-batch, (N, x_dim(s))
        :param K: Samples per observation
        :return: IS estimate for each point in X
        """
        x = None
        MB = batch[0].shape[0]
        elbo = torch.zeros(K, MB)
        logs = {'nll_z_L': 0,}
        for k in range(K):
            fwd_output = self.forward(batch, mode='test')
            _, p_xz, z_sample, q_dist, p_dist = fwd_output
            if x is None:
                x = batch[0]
                if isinstance(p_xz, Bernoulli):
                    x = torch.bernoulli(0.5 * (x + 1))
            re = - p_xz.log_prob(x)  # MB
            # KL-divergence (L values)
            kl, kl_logs = self.kl(fwd_output)
            kl = kl.sum(1)
            assert re.shape == kl.shape
            elbo[k] = - re - kl
            logs['nll_z_L'] += kl_logs['nll_z_L_prior'] * MB / K

        ll = torch.logsumexp(elbo, 0) - np.log(K)
        return -ll, logs

    def ladder_reconstructions(self, encoder_s, im):
        N_z = len(encoder_s)
        n_sample = 5
        rows = []
        for z_num in range(0, N_z, self.get_freq(N_z)):
            q_curr = [s_e[im:im+1].repeat(n_sample, 1, 1, 1) for s_e in encoder_s.copy()]
            # create [0, 0, .., q_{N-z_num}, ..., q_L]
            for i in range(N_z - z_num - 1):
                q_curr[i] = None
            p_xz_params, _, _, z_samples = self.decoder(q_curr, t=0.8)
            p_xz = self.likelihood(*p_xz_params)
            x = self.get_x_from_px(p_xz)
            rows.append(x)
        return torchvision.utils.make_grid(torch.cat(rows, 0), nrow=n_sample,
                                           normalize=True, scale_each=True)

    def get_freq(self, N_z):
        if N_z < 10:
            return 1
        elif N_z <= 20:
            return 2
        elif N_z <= 40:
            return 4
        elif N_z <= 100:
            return 10
        return N_z // 10

    def val_pics(self, batch, fwd_output):
        logs = {}
        encoder_s = self.encode(batch)

        for im in range(4):
            logs[f'ladder/h_{im}'] = wandb.Image(
                self.ladder_reconstructions(encoder_s, im)
            )
        logs.update(self.visualize_z_L_prior(batch, fwd_output))
        # logs.update(self.calculate_collapsed_dim(fwd_output))
        return logs

    def process_z_L_samples(self, z_L):
        return z_L[:, :3]

    def visualize_z_L_prior(self, batch, fwd_output):
        logs = {}

        # prior samples
        s_dec = torch.ones((8,) + self.decoder.s_L_shape, device=self.decoder.device)
        _, _, z_L_sample, _ = self.decoder.decoder_blocks[0](None, s_dec=s_dec, mode='val', t=1.0)
        logs[f'z_L_prior/sample'] = wandb.Image(self.process_z_L_samples(z_L_sample).data.cpu())

        if isinstance(self.decoder.z_L_prior, DiffusionPrior):
            ddgm = self.decoder.z_L_prior

            logs[f'z_L_prior/gamma_min'] = ddgm.noise_schedule.gamma_min.detach().cpu()
            logs[f'z_L_prior/gamma_max'] = ddgm.noise_schedule.gamma_max.detach().cpu()

            # compute loss for each step
            z_L = fwd_output[2][0]
            MB = z_L.shape[0]
            ss = max(ddgm.T//25, 1)
            
            gamma_0 = ddgm.noise_schedule.gamma_min.repeat(MB, 1)
            alpha_0 = ddgm.alpha(gamma_0)
            sigma_0 = ddgm.sigma2(gamma_0).pow(0.5)
            losses = [ddgm.rec_loss(z_L, alpha_0 * z_L + sigma_0 * torch.randn_like(z_L), gamma_0).cpu().sum(1)]

            for t in range(1, ddgm.T+1, ss):
                timestamps = torch.ones(z_L.shape[0], 1).long().to(batch[0].device) * t    / ddgm.T
                ts_prev = timestamps - (1. / ddgm.T)
                w, gamma_t, gamma_prev = ddgm.compute_mse_weight(timestamps, ts_prev)
                eps = torch.randn_like(z_L)
                z_t = ddgm.alpha(gamma_t) * z_L + ddgm.sigma2(gamma_t).pow(0.5) * eps
                eps_hat = ddgm.predict_eps(z_t, timestamps, gamma_t)
                L_t, mse = ddgm.sample_L_t(eps, eps_hat, w)
                losses.append(L_t.cpu().sum(-1) / ddgm.T)
            losses.append(ddgm.latent_loss(z_L).sum(1).cpu())
                
            
            losses = torch.stack(losses, 1)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
            ax.errorbar(list(range(1, ddgm.T+1, ss)), losses.mean(0)[1:-1], yerr=losses.std(0)[1:-1],
                        label='Val. log_prob')
            ax.legend()
            ax.grid(True)
            logs[f'z_L_prior/ddgm_loss'] = wandb.Image(fig)
            plt.close()

            logs[f'z_L_prior/L0'] = losses.mean(0)[0].item()
            logs[f'z_L_prior/LT'] = losses.mean(0)[-1].item()

            #  plot ddgm reconstrution loss alogn with log q part:
            # E_{q(u|f(x))} [\log q(u | f(x)) - E_{r(y_1|u)} \log p(u | y_1)]
            log_q = fwd_output[3][0].log_prob(z_L).mean(0)
            log_p = -logs[f'z_L_prior/L0'] 
            logs[f'z_L_prior/Elog_q + L0'] = log_q - log_p

            # plot 'reconstructions' for each step
            freq = self.get_freq(ddgm.T)
            q_samples = [self.process_z_L_samples(z_L).cpu()]
            reconstruction = [self.process_z_L_samples(z_L).cpu()]
            all_steps = range(1, ddgm.T+1)

                
            for t in all_steps:
                timestamps = torch.ones(z_L.shape[0], 1).long().to(z_L.device) * t / ddgm.T
                z_t = ddgm.q_sample(z_L, timestamps)
                z_inv = z_t.clone()
                if t % freq == 0:
                    for t_inv in all_steps[:t + 1][::-1]:
                        t = torch.ones(z_L.shape[0], 1).long().to(z_L.device) * t_inv / ddgm.T
                        s = t - 1 / ddgm.T
                        z_inv = ddgm.sample_p(z_inv, t, s, temp=1.)
                    
                    reconstruction.append(self.process_z_L_samples(z_inv).cpu())

                q_samples.append(
                    self.process_z_L_samples(z_t).cpu()
                )

            T = ddgm.T + 1
            res = []
            for i in range(4):
                res += [q_samples[t][i:i + 1] for t in range(0, T, freq)]
            zs_inv = torchvision.utils.make_grid(torch.cat(res, 0),
                                                 nrow=len(range(0, T, freq)),
                                                 normalize=True, scale_each=False)
            logs[f'z_L_prior/ddgm_q_samples'] = wandb.Image(zs_inv)


            res = []
            for i in range(4):
                res += [reconstruction[t][i:i + 1] for t in range(len(reconstruction))]
            ys_inv = torchvision.utils.make_grid(torch.cat(res, 0),
                                                 nrow=len(reconstruction),
                                                 normalize=True, scale_each=False)
            logs[f'z_L_prior/ddgm_reconstructions'] = wandb.Image(ys_inv)
        return logs