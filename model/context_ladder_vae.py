import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
import os
import torch.nn as nn
import torch.nn.functional as F
from model.vae import LADDER_VAE, compute_sample_kl


class CTX_LADDER_VAE(LADDER_VAE):
    def __init__(self,
                 encoder,
                 decoder,
                 likelihood,
                 beta_start,
                 beta_end,
                 warmup,
                 is_k,
                 latent_scales,
                #  free_bits_thr,
                 pretrain_prior_epochs:int=0,
                 **kwargs):
        super().__init__(encoder,
                         decoder,
                         likelihood,
                         beta_start,
                         beta_end,
                         warmup,
                         is_k,
                         latent_scales,
                        #  free_bits_thr
                         )
        self.pretrain_prior_epochs=pretrain_prior_epochs

    def encode(self, batch):
        x = batch[0]
        y = self.decoder.decoder_blocks[0].x_to_ctx(x, preprocess=False)
        
        if self.current_epoch is None or self.current_epoch >= self.pretrain_prior_epochs:
            encoder_s = self.encoder(x)
        else:
            encoder_s = [None] * self.decoder.num_latents
        return encoder_s + [y]

    def generate_x(self, N=25, t=None):
        enc_s = [None for _ in range(sum(self.latent_scales)+1)]
        p_xz_params, _, _, _ = self.decoder(enc_s,  mode='sample', N=N, t=t)
        p_xz = self.likelihood(*p_xz_params)
        return self.get_x_from_px(p_xz)

    def process_z_L_samples(self, z_L):
        return self.decoder.decoder_blocks[0].ctx_to_x(z_L)
    
    def compute_loss(self, batch, fwd_output, beta):
        if self.current_epoch is None or self.current_epoch >= self.pretrain_prior_epochs:
            return super().compute_loss(batch, fwd_output, beta)
        else:
            logs = {}
            _, _, z_samples, q_dist, p_dist = fwd_output
            bpd_coef = 1. / np.log(2.) / np.prod(batch[0].shape[1:])
            
            log_q = q_dist[0].log_prob(z_samples[0], reduce_dim=True).mean(0)
            nll_ddgm = - p_dist[0]
            nll_dgm = nll_ddgm.reshape(nll_ddgm.shape[0], -1).sum(1).mean(0)

            loss = (log_q + nll_dgm) * bpd_coef
            
            zL_dim = np.prod(z_samples[0].shape[1:])
            logs = {
                'kl_L':(log_q + nll_dgm).data / zL_dim,
                'nll_z_L_prior': nll_dgm.data,
                'reconstruction': 0,
                'kl': 0,
                'loss': 0,
                'loss_per_pixel': 0,
            }
            return loss, logs
        
    def calculate_active_units(self, fwd_output):
        _, _, z_sample, q_dist, p_dist = fwd_output
        MB = z_sample[0].shape[0]
        logs = {}
        # AU as in Burda'15 (Cov_x E[z])
        E_q = [q_dist[i].get_E().reshape(MB, -1).data.cpu() for i in range(1, len(q_dist))]
        unit_cnt = sum([mu.shape[1] for mu in E_q])
        cov = [torch.var(mu, dim=0).float() for mu in E_q]
        logs['misc/Active units (log Cov)'] = [torch.log(torch.clamp(c, 1e-10)) for c in cov]
        logs['misc/% Active units (Cov > 0.01)'] = sum(torch.cat(cov) > 0.01)/unit_cnt
        logs['misc/% Active units (Cov > 0.001)'] = sum(torch.cat(cov) > 0.001)/unit_cnt
        return logs