import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional

from utils.distribution import Normal, approx_standard_normal_cdf


class DiffusionPrior(nn.Module): 
    def __init__(self,
                 model,
                 T,
                 noise_schedule,
                 t_sample='uniform',
                 parametrization='x',
                 ll='discretized_gaussian',
                 cont_time=True,
                 ):
        super(DiffusionPrior, self).__init__()
        self.model = model
        self.device = 'cuda'
        self.T = T
        self.t_sample = t_sample
        self.parametrization = parametrization
        self.cont_time = cont_time
        self.ll = ll
        self.noise_schedule = noise_schedule 
        self.num_vals = 256

    def alpha(self, gamma):
        alpha = torch.sqrt(torch.sigmoid(-gamma))
        return alpha[..., None, None]

    def sigma2(self, gamma):
        return torch.sigmoid(gamma)[..., None, None]
    
    def gamma_jvp(self, ts, return_gamma=False):
        gamma_t, jvp = torch.autograd.functional.jvp(func=self.noise_schedule,
                                                     inputs=ts,
                                                     v=torch.ones_like(ts),
                                                     create_graph=True)
        if return_gamma:
            return jvp, gamma_t
        return jvp
    
    def log_snr(self, gamma):
        return -gamma

    def forward(self, x):
        MB = x.shape[0]
        if self.device is None:
            self.device = x.device
        # Get z_0 from x
        gamma_0 = self.noise_schedule.gamma_min.repeat(MB, 1)
        eps_0 = torch.randn_like(x)
        alpha_0 = self.alpha(gamma_0)
        sigma_0 = self.sigma2(gamma_0).pow(0.5)
        z_0 = alpha_0 * x + sigma_0 * eps_0
        # sample t
        ts, ts_prev = self.sample_t(batch_size=MB, device=x.device)
        w, gamma_t, gamma_prev = self.compute_mse_weight(ts, ts_prev)

        # sample z_t ~ q(z_t | x)
        eps = torch.randn_like(x)
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma2(gamma_t).pow(0.5)
        z_t = alpha_t * x + sigma_t * eps

        # predict the noise
        eps_hat = self.predict_eps(z_t, ts, gamma_t)
        return {
            'z_0': z_0,
            'z_t': z_t,
            'eps_0': eps_0,
            'eps': eps,
            'eps_hat': eps_hat,
            'w': w,
            'ts': ts,
            'ts_prev': ts_prev,
            'gamma_t': gamma_t,
            'gamma_prev': gamma_prev,
            'gamma_0': gamma_0,
        }
    

    def log_prob(self, x, mode='train', reduce_dim=True):
        """
        :param z_0: (MB, ch, h, w)
        :param mode: 'train', 'test' or 'val'
        :return: log p(z_0)
        """
        assert mode in ['train', 'val', 'test']
        fwd_output = self.forward(x)

        # reconstruction loss -log p(x|z_0)
        L_0 = self.rec_loss(x, fwd_output['z_0'], fwd_output['gamma_0'])

        # mse loss
        eps, eps_hat, w = fwd_output['eps'], fwd_output['eps_hat'], fwd_output['w']
        if mode == 'test':
            L_full, mse_full = self.full_L_t(x)
            L_t = L_full.mean(1)
        else:
            L_t, mse = self.sample_L_t(eps, eps_hat, w)

        L_T = self.latent_loss(x)
        neg_log_p =  (L_0 + L_t + L_T)
        return -neg_log_p

    def sample_t(self, batch_size, device):
        if self.t_sample == 'uniform':
            ts = torch.rand((batch_size, 1), device=device)
        elif self.t_sample == 'uniform_stable':
            u_rand = torch.rand((1,))
            step = 1 / batch_size
            unif = torch.arange(0., 1., step)
            ts = torch.clamp(torch.fmod(u_rand + unif, 1), 1e-5, 1-1e-5)
            ts = ts.reshape(batch_size, 1)
            ts = ts.to(device)
        else:
            NotImplementedError(f"unknown t sampling schedule: {self.t_sample}")
        # discretize if not continious time
        if not self.cont_time:
            ts = torch.ceil(ts * self.T) / self.T
            ts_prev = ts - (1. / self.T)
        else:
            ts_prev = None
        return ts, ts_prev

    def compute_mse_weight(self, ts, ts_prev):
        if self.cont_time:
            w, gamma_t = self.gamma_jvp(ts, return_gamma=True)
            w = 0.5 * w
            gamma_prev = None
        else:
            gamma_t = self.noise_schedule(ts)
            gamma_prev = self.noise_schedule(ts_prev)
            log_snr_t = self.log_snr(gamma_t)
            log_snr_prev = self.log_snr(gamma_prev)
            w = 0.5 * self.T * torch.expm1(log_snr_prev - log_snr_t)
        return w, gamma_t, gamma_prev

    def rec_loss(self, x, z_0, gamma_0):
        alpha_0 = self.alpha(gamma_0)
        var_0 = self.sigma2(gamma_0)
        sigma_0 = var_0.pow(0.5)
        z_0_ch = z_0.shape[1]

        if self.ll == 'gaussian':
            log_var = -self.log_snr(gamma_0)
            mu = (z_0 / alpha_0).reshape(x.shape[0], -1)
            p_dist = Normal(mu, log_var.reshape(x.shape[0], -1))
            rec_ll = p_dist.log_prob(x.reshape(x.shape[0], -1), reduce_dim=False)
        else:
            # create tensor of all possible discrete values
            x_vals = torch.arange(0, self.num_vals)[:, None]  # (256, 1)
            x_vals = x_vals.repeat(1, z_0_ch)  # (256, 3)
            # (256, 3) mapped to [-1, 1]
            x_vals = (2 * x_vals + 1.0) / self.num_vals - 1
            x_vals = x_vals.transpose(1, 0)[None, None, None, :, :]  # (1, 1, 1, 3, 256)
            x_vals = x_vals.swapaxes(2, 3).swapaxes(1, 2)  # (1, 3, 1, 1, 256)
            x_vals = x_vals.to(z_0.device)
            inv_stdv =  self.num_vals * 9

            if self.ll == 'discretized_gaussian':
                # parameters of the gaussian
                centered_x_vals = z_0[..., None] - x_vals  # (MB, ch, h, w, 256)
                # calculate CDF (k + 1/255)
                plus_in = inv_stdv * (centered_x_vals + 1.0 / (self.num_vals - 1))
                cdf_plus = approx_standard_normal_cdf(plus_in)

                # calculate CDF (k - 1/255)
                min_in = inv_stdv * (centered_x_vals - 1.0 / (self.num_vals - 1))
                cdf_min = approx_standard_normal_cdf(min_in)
                logprobs = torch.log((cdf_plus - cdf_min).clamp(min=1e-10))
            elif self.ll == 'vdm':
                # calculate lop q(z_0|x=i), i=1...2**bits
                inv_var = 1 / var_0
                logits = - 0.5 * (z_0[..., None] - alpha_0[..., None] * x_vals) ** 2 * inv_var[
                    ..., None]
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

            # create x OHE
            x = (torch.clamp(x, -1., 1.) + 1.) / 2. * (self.num_vals - 1)
            x = x.round().long() 
            x_onehot = torch.nn.functional.one_hot(x, num_classes=self.num_vals)
            # compute log prob
            rec_ll = torch.sum(x_onehot * logprobs, axis=-1).reshape(x.shape[0], -1) #.sum(1)
        return -rec_ll

    def sample_L_t(self, eps, eps_hat, w):
        assert len(w.shape) == 2, f'w should be 2D, got {w.shape}'
        # mse loss
        mse = (eps_hat - eps).pow(2)
        L_t = w[..., None, None] * mse

        MB = eps.shape[0]
        L_t = L_t.reshape(MB, -1)
        mse = mse.reshape(MB, -1).sum(1)
        return L_t, mse

    def full_L_t(self, x):
        MB = x.shape[0]
        all_Lts = []
        all_mses = []

        all_ts = (torch.arange(1, self.T+1) / self.T).reshape(self.T, 1, 1).repeat(1, MB, 1)        
        all_ts = all_ts.to(x.device)

        for ts in all_ts:
            ts_prev = ts - (1. / self.T)
            w, gamma_t, gamma_prev = self.compute_mse_weight(ts, ts_prev)

            # sample z_t ~ q(z_t | x) and predict the noise
            eps = torch.randn_like(x)
            alpha_t = self.alpha(gamma_t)
            sigma_t = self.sigma2(gamma_t).pow(0.5)
            z_t = alpha_t * x + sigma_t * eps
            eps_hat = self.predict_eps(z_t, ts, gamma_t)

            L_curr, mse_curr = self.sample_L_t(eps, eps_hat, w)
            all_Lts.append(L_curr)
            all_mses.append(mse_curr)
        L_full = torch.stack(all_Lts, dim=1)
        mse_full = torch.stack(all_mses, dim=1)
        return L_full, mse_full

    def latent_loss(self, x):
        MB = x.shape[0]
        gamma_T = self.noise_schedule.gamma_max.repeat(MB, 1)
        var_T = self.sigma2(gamma_T)
        alpha_T = self.alpha(gamma_T)
        mu_T_sq = (alpha_T * x).pow(2)
        latent_loss = 0.5 * (mu_T_sq + var_T - torch.log(var_T) - 1.)
        latent_loss = latent_loss.reshape(MB, -1) 
        return latent_loss
    
    def predict_eps(self, z_t, ts, gamma_t):
        unet_out = self.model(z_t, ts.flatten())
        if self.parametrization == 'eps':
            return unet_out
        elif self.parametrization == 'v':
            alpha_t = self.alpha(gamma_t)
            sigma2_t = self.sigma2(gamma_t)
            sigma_t = sigma2_t.pow(0.5)
            eps_hat = (sigma_t * z_t + alpha_t * unet_out) / (sigma2_t + alpha_t.pow(2))
            return eps_hat
        else:
            raise NotImplementedError(f"unknown parametrization: {self.parametrization}")

    def eval_is_ll(self, z_0, is_k=1):
        """
        Importance sampling estimation of the NLL
        :param z_0: batch of data points
        :param is_k: number of importance samples
        :return:
        """
        MB = z_0.shape[0]
        elbo = torch.zeros(is_k, z_0.shape[0], device=z_0.device)
        for k in range(is_k):
            elbo[k] = self.log_prob(z_0, mode='test').sum(1)
        ll = torch.logsumexp(elbo, 0) - np.log(is_k)
        return ll

    def sample(self, N, t=1.):
        '''
        t stands for temperature.
        '''
        shape = [N, self.model.raw_in_channels,  self.model.image_size, self.model.image_size]
        z_curr = torch.randn(*shape, device=self.device)
        # loop through i_s = [T-1, T-2, ..., 0]; i_t = [T, T-1, ..., 1]
        #    to sample from p(z_s | z_t), where t > s
        all_ts  = (torch.Tensor(range(self.T, 0, -1)) / self.T).unsqueeze(1).repeat(1, N).unsqueeze(2)  # T x N x 1
        prev_ts = all_ts - (1/self.T)
        all_ts = all_ts.to(z_curr.device)
        prev_ts = prev_ts.to(z_curr.device)
        with torch.no_grad():
            for i in range(self.T):
                ts = all_ts[i]
                ts_prev = prev_ts[i]
                # sample z_s | z_t, where t > s
                z_curr = self.sample_p(z_t=z_curr,
                                       t=ts,
                                       s=ts_prev,
                                       temp=t)
        return z_curr

    def sample_p(self, z_t, t, s, temp=1.):
        assert torch.all(t > s), f'Expect t > s, got t: {t.data}, s: {s.data}'
        gamma_t = self.noise_schedule(t)
        gamma_s = self.noise_schedule(s)
        x_pred = self.endpoint_estimate(z_t, gamma_t, t)
        z_s = self.sample_q_posterior(z_t, x_pred, gamma_t, gamma_s, temp, eps=None)
        return z_s

    def endpoint_estimate(self, z_t, gamma_t, t):
        alpha_t = self.alpha(gamma_t)
        sigma2_t = self.sigma2(gamma_t)
        sigma_t = sigma2_t.pow(0.5)

        if self.parametrization == 'eps':
            eps_hat = self.model(z_t, t.flatten())
            x_pred = (z_t - sigma_t * eps_hat) / alpha_t
        elif self.parametrization == 'v':
            v_hat = self.model(z_t, t.flatten())
            x_pred = (alpha_t * z_t - sigma_t * v_hat) / (sigma2_t + alpha_t.pow(2))
        x_pred = torch.clamp(x_pred, -1., 1.)
        return x_pred
        
    def q_sample(self, z_0, t):
        eps = torch.randn_like(z_0)
        gamma_t = self.noise_schedule(t)
        return self.alpha(gamma_t) * z_0 + self.sigma2(gamma_t).pow(0.5) * eps

    def sample_q_posterior(self, z_t, x_pred, gamma_t, gamma_s, temp=1., eps=None):
        alpha_t = self.alpha(gamma_t)
        alpha_s = self.alpha(gamma_s)
        alpha_ts = alpha_t / alpha_s

        sigma_sq_t = self.sigma2(gamma_t)
        sigma_sq_s = self.sigma2(gamma_s)
        sigma_sq_ts = sigma_sq_t - alpha_ts**2 * sigma_sq_s

        mu_zt_coef = alpha_ts * sigma_sq_s / sigma_sq_t
        mu_x_coef = alpha_s * sigma_sq_ts / sigma_sq_t
        sigma_coef = (sigma_sq_ts * sigma_sq_s / sigma_sq_t).pow(0.5)
        
        mu = mu_zt_coef * z_t + mu_x_coef * x_pred
        if eps is None:
            eps = torch.randn_like(z_t)
        if temp is None:
            temp = 1.
        z_s = mu + temp * sigma_coef * eps
        return z_s


def _linear(min, max, t):
    return min + (max - min) * t


def _cosine(min, max, t):
    t_max = torch.atan(torch.exp(0.5 * max))
    t_min = torch.atan(torch.exp(0.5 * min))
    return 2 * torch.log(torch.tan(_linear(t_min, t_max, t)))


class LinearNoiseSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max, train=True):
        super().__init__()
        self.gamma_min = nn.Parameter(torch.tensor(float(gamma_min)).reshape(1, 1),
                                        requires_grad=train)
        self.gamma_max = nn.Parameter(torch.tensor(float(gamma_max)).reshape(1, 1),
                                        requires_grad=False)

    def forward(self, ts):
        # assert torch.all(ts <= 1.) and torch.all(ts >= 0.)
        ts = torch.clamp(ts, 0., 1.)
        return _linear(self.gamma_min, self.gamma_max, ts)

    def scale_output(self, gamma):
        MB, ch = gamma.shape
        dummy_0 = self.forward(torch.zeros(MB, 1, device=gamma.device))
        dummy_1 = self.forward(torch.ones(MB, 1,  device=gamma.device))
        trg_range = self.gamma_max - self.gamma_min
        curr_range = dummy_1 - dummy_0
        return self.gamma_min + trg_range * (gamma - dummy_0) / curr_range


class CosineNoiseSchedule(LinearNoiseSchedule):
    def __init__(self, gamma_min, gamma_max, train=True):
        super().__init__(gamma_min, gamma_max, train)

    def forward(self, ts):
        assert torch.all(ts <= 1.) and torch.all(ts >= 0.)
        return _cosine(self.gamma_min, self.gamma_max, ts)


def discretized_gaussian_log_likelihood(x, *, means, log_scales, n_bits = 5):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities
    """
    bins = 2 ** n_bits - 1
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / bins )
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / bins)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-10))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-10))
    log_cdf_delta = torch.log((cdf_plus - cdf_min).clamp(min=1e-10))
    log_probs = torch.where(
        x <= -1. + 1./bins,
        log_cdf_plus,
        torch.where(x >= 1. - 1./bins,
                    log_one_minus_cdf_min,
                    log_cdf_delta),
    )
    assert log_probs.shape == x.shape
    return log_probs
