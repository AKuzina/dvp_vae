import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Distribution(nn.Module):
    def __init__(self):
        super(Distribution, self).__init__()

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, N=1, t=None):
        raise NotImplementedError


class Normal(Distribution):
    def __init__(self, mu, log_var, *args, **kwargs):
        super(Normal, self).__init__()
        self.mu = mu
        self.log_var = log_var

    def log_prob(self, x, reduce_dim=True):
        MB = x.shape[0]
        if len(x.shape) > len(self.mu.shape):
            MB = x.shape[0]
        log_p = -0.5 * (math.log(2.0*math.pi) +
                        self.log_var +
                        torch.pow(x - self.mu, 2) / (torch.exp(self.log_var) + 1e-10))
        if reduce_dim:
            return log_p.reshape(MB, -1).sum(1)
        else:
            return log_p.reshape(MB, -1)

    def sample(self, N=None, t=None):
        size = self.mu.shape
        if N is not None:
            size = torch.Size([N]) + size
        z_sample = torch.empty(size, device=self.mu.device)
        if t is not None:
            sigma = (0.5 * (self.log_var + torch.ones_like(self.log_var) * math.log(t))).exp()
        else:
            sigma = (0.5*self.log_var).exp()
        eps = z_sample.normal_()
        return self.mu + sigma*eps

    def update(self, delta_mu, delta_logvar):
        self.mu = self.mu + delta_mu
        self.log_var = self.log_var + delta_logvar

    def get_E(self):
        return self.mu

    def entropy(self):
        c = 1 + math.log(math.pi*2)
        return 0.5 * (c + self.log_var).sum()

    def kl(self, dist):
        """
        compute kl-divergence with the given distribution
        """
        assert isinstance(dist, Normal), 'Can only compute analytical kl for gaussians'
        log_v_r = dist.log_var - self.log_var
        mu_r_sq = (self.mu - dist.mu) ** 2
        kl = 0.5 * (-1 + log_v_r + (self.log_var.exp() + mu_r_sq) / dist.log_var.exp())
        return kl
    
class NormalMixture(Distribution):
    def __init__(self, mu, log_var, *args, **kwargs):
        super().__init__()
        self.mu = mu
        self.log_var = log_var

    def log_prob(self, x, reduce_dim=True):
        MB = x.shape[0] # assume first dimention of x is batch and first dimention of mu is num_components
        K = self.mu.shape[0]
        assert len(x.shape) == len(self.mu.shape)
        # x = x.unsqueeze(1) # MB x 1 x dims
        mu = self.mu.unsqueeze(0) # 1 x K x dims
        log_var = self.log_var.unsqueeze(0) # 1 x K x dims
        
        log_ps = -0.5 * (math.log(2.0*math.pi) +
                        log_var +
                        torch.pow(x.unsqueeze(1) - mu, 2) / (torch.exp(log_var) + 1e-10)) # MB x K x dims
            
        log_p = torch.logsumexp(log_ps, dim=1) - math.log(K) # MB x dims
        assert log_p.shape == x.shape

        if reduce_dim:
            return log_p.reshape(MB, -1).sum(1)
        else:
            return log_p.reshape(MB, -1)

    def sample(self, N=None, t=None):
        K = self.mu.shape[0]
        # sample components ids
        mixture_idx = np.random.choice(K, size=N, replace=True)
        mu = self.mu[mixture_idx]
        logvar = self.log_var[mixture_idx]

        size = torch.Size([N]) + self.mu.shape[1:]
        z_sample = torch.empty(size, device=self.mu.device)
        if t is not None:
            sigma = (0.5 * (logvar + torch.ones_like(logvar) * math.log(t))).exp()
        else:
            sigma = (0.5*logvar).exp()
        eps = z_sample.normal_()
        return mu + sigma * eps

    def get_E(self):
        return self.mu

    def entropy(self):
        pass


CONST_LOG_INV_SQRT_2PI = math.log( 1 / math.sqrt(2 * math.pi))


class TruncatedNormal(Distribution):
    def __init__(self, mu, log_var, *args, **kwargs):
        """
        Nornal distribution truncated to [-1, 1]
        Adapted from https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
        """
        super().__init__()
        self.mu = mu
        self.log_var = log_var
        scale = torch.exp(0.5 * self.log_var)

        self.a = (-torch.ones_like(self.mu)- self.mu) / scale
        self.b = (torch.ones_like(self.mu) - self.mu) / scale
        self.eps = torch.finfo(self.a.dtype).eps
        
        self.phi_a = self._big_phi(self.a)
        self.phi_b = self._big_phi(self.b)
        self.Z = (self.phi_b - self.phi_a).clamp_min(self.eps)
        self.log_Z = self.Z.log()

        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self.std_mean = -(self._little_phi_b - self._little_phi_a) / self.Z

    def log_prob(self, x, reduce_dim=True):
        MB = x.shape[0]
        if len(x.shape) > len(self.mu.shape):
            MB = x.shape[0]
        scale = torch.exp(0.5 * self.log_var)
        value = (x - self.mu) / scale


        log_p = CONST_LOG_INV_SQRT_2PI - self.log_Z - 0.5 * self.log_var - 0.5 * (value ** 2) 
        if reduce_dim:
            return log_p.reshape(MB, -1).sum(1)
        else:
            return log_p.reshape(MB, -1)
        
    @staticmethod
    def _inv_big_phi(x):
        return math.sqrt(2) * (2 * x - 1).erfinv()
    
    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x / math.sqrt(2)).erf())
    
    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() / math.sqrt(2 * math.pi)

    def sample(self, N=None, t=None):
        size = self.mu.shape
        if N is not None:
            size = torch.Size([N]) + size
        U = torch.empty(size, device=self.mu.device).uniform_(self.eps, 1-self.eps)    
        
        if t is not None:
            sigma = (0.5 * (self.log_var + torch.ones_like(self.log_var) * math.log(t))).exp()
        else:
            sigma = (0.5*self.log_var).exp()
        std_sample =  self._inv_big_phi(self.phi_a + U * self.Z)    
        sample = std_sample * sigma + self.mu
        return sample
    
    def get_E(self):
        return self.std_mean  * torch.exp(0.5 * self.log_var) + self.mu


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))



def create_standard_normal_prior(size):
    size = list(size)
    mu = nn.Parameter(torch.zeros(size), requires_grad=False)
    logvar = nn.Parameter(torch.zeros(size), requires_grad=False)
    return Normal(mu, logvar)


def create_gaussian_prior(size):
    size = list(size)
    mu = nn.Parameter(torch.zeros(size), requires_grad=True)
    logvar = nn.Parameter(torch.randn(size)*0.01, requires_grad=True)
    return Normal(mu, logvar)


def create_MoG_prior(size, num_components):
    size = [1, 1] + list(size)
    size[1] *= num_components
    mu = nn.Parameter(torch.zeros(size), requires_grad=True)
    logvar = nn.Parameter(torch.randn(size)*0.01, requires_grad=True)
    return MixtureOfGaussians(means=mu, logvars=logvar, num_comp=num_components)


class Delta(Distribution):
    def __init__(self, x):
        self.x = x

    def log_prob(self, x, reduce_dim=True):
        out =  torch.zeros(x.shape, device=x.device).reshape(x.shape[0], -1)
        if reduce_dim:
            out = out.sum(1)
        return out

    def sample(self, N=None):
        x_sample = self.x.clone()
        if N is not None:
            size = torch.Size([N]) + self.x.size()
            x_sample = x_sample.unsqueeze(0).repeate(size)
        return x_sample

    def get_E(self):
        return self.x


class Bernoulli(Distribution):
    def __init__(self, p, *args, **kwargs):
        super(Bernoulli, self).__init__()
        eps = 1e-7
        self.p = torch.clamp(p, min=eps, max=1-eps)

    def log_prob(self, x):
        MB = x.shape[0]
        assert torch.max(x).item() <= 1.0 and torch.min(x).item() >= 0.0
        log_p = x * torch.log(self.p) + (1. - x) * torch.log(1. - self.p)
        return log_p.reshape(MB, -1).sum(1)

    def sample(self, N=None):
        p = self.p
        if N is not None:
            p = p.unsqueeze(0).repeat([N] + [1 for _ in range(len(p.shape))])
        return torch.bernoulli(p)

    def get_E(self):
        return self.p


class Logistic256(Distribution):
    def __init__(self, mean, var, *args, **kwargs):
        super(Logistic256, self).__init__()
        self.mean = mean
        softplus = nn.Softplus(0.4)
        self.log_var = torch.log(softplus(torch.clamp(var, min=-20.)))

    def log_prob(self, x, low_bit=False):
        assert x.min() >= -1. and x.max() <= 1.
        # rescale x to [-1, 1] if needed
        if x.min() >= 0:
            x = 2. * x - 1

        if low_bit:
            max_bit = 31.
        else:
            max_bit = 255.

        centered = x - self.mean  # B, C, H, W
        inv_stdv = torch.exp(- self.log_var)

        # each pixel has a bin of width 2/n_bit -> half of the bin is 1/n_bit
        plus_in = inv_stdv * (centered + 1. / max_bit)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_stdv * (centered - 1. / max_bit)
        cdf_min = torch.sigmoid(min_in)

        # probability to be in the bin
        # cdf_delta = cdf_plus - cdf_min
        cdf_delta = torch.clamp(cdf_plus - cdf_min, min=1e-10)
        log_probs = torch.log(cdf_delta)

        # for pixel 0 we have -\inf instead of min_in
        log_cdf_plus = plus_in - F.softplus(plus_in)
        pix_0 = -1. + 1./max_bit
        log_probs = torch.where(x <= pix_0,
                                log_cdf_plus,
                                log_probs)

        # for pixel 255 we have \inf instead of plus_in
        log_one_minus_cdf_min = -F.softplus(min_in)
        pix_255 = 1. - 1./max_bit
        log_probs = torch.where(x >= pix_255,
                                log_one_minus_cdf_min,
                                log_probs)
        log_probs = log_probs.sum(dim=[1, 2, 3])  # MB
        return log_probs

    def sample(self, N=None, t=None):
        size = self.mean.shape
        if N is not None:
            size = torch.Size([N]) + size
        u = torch.Tensor(size).uniform_(1e-5, 1. - 1e-5)
        u = u.to(self.mean.device)
        if t is not None:
            scale = torch.exp(self.log_var + torch.ones_like(self.log_var) * math.log(t))
        else:
            scale = torch.exp(self.log_var)
        x = self.mean + scale * (torch.log(u) - torch.log(1. - u))
        return x

    def get_E(self):
        return self.mean

    def entropy(self):
        return self.logvar + 2


class MixtureLogistic256(Distribution):
    # Using the implementations from
    # https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/losses.py
    # https://github.com/openai/vdvae/blob/ea35b490313bc33e7f8ac63dd8132f3cc1a729b4/vae_helpers.py
    def __init__(self, logit_probs, mean, log_var, coeffs, low_bit=False):
        super(MixtureLogistic256, self).__init__()
        self.low_bit = low_bit
        self.logit_probs = logit_probs  # MB, M, H, W
        self.data_ch = 3
        mb, self.num_mix, h, w = logit_probs.shape
        self.means = mean  # MB, 3, M, H, W
        # softplus = nn.Softplus(0.4)
        # self.log_var = torch.log(softplus(torch.clamp(log_var, min=-8.))) # MB, 3, M, H, W
        self.log_var = torch.clamp(log_var, min=-8., max=1.) # MB, 3, M, H, W
        self.coeffs = torch.tanh(coeffs)

    def log_prob(self, x):

        assert x.min() >= -1. and x.max() <= 1.
        # rescale x to [-1, 1] if needed
        if x.min() >= 0:
            x = 2. * x - 1

        if self.low_bit:
            max_bit = 31.
        else:
            max_bit = 255.

        x = x.unsqueeze(2)  # MB, 3, 1, H, W

        # RGB AR
        mean1 = self.means[:, 0:1, :, :, :]  # B, 1, M, H, W
        mean2 = self.means[:, 1:2, :, :, :] + self.coeffs[:, 0:1, :, :, :] * x[:, 0:1, :, :, :]  # B, 1, M, H, W
        mean3 = self.means[:, 2:3, :, :, :] + self.coeffs[:, 1:2, :, :, :] * x[:, 0:1, :, :, :] + self.coeffs[:, 2:3, :, :,:] * x[:, 1:2,:, :,:]  # B, 1, M, H, W
        means = torch.cat([mean1, mean2, mean3], dim=1)  # B, C, M, H, W

        # centered_x = x - self.means  # B, C, M, H, W
        centered_x = x - means  # B, C, M, H, W

        inv_stdv = torch.exp(-self.log_var)

        # each pixel has a bin of width 2/n_bit -> half of the bin is 1/n_bit
        plus_in = inv_stdv * (centered_x + 1. / max_bit)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_stdv * (centered_x - 1. / max_bit)
        cdf_min = torch.sigmoid(min_in)

        # probability to be in the bin
        cdf_delta = torch.clamp(cdf_plus - cdf_min, min=1e-10)
        log_probs = torch.log(cdf_delta)

        # for pixel 0 we have -\inf instead of min_in
        log_cdf_plus = plus_in - F.softplus(plus_in)
        pix_0 = -1. + 1./max_bit
        log_probs = torch.where(x.repeat(1, 1, self.num_mix, 1, 1) <= pix_0,
                                log_cdf_plus,
                                log_probs)

        # for pixel 255 we have \inf instead of plus_in
        log_one_minus_cdf_min = -F.softplus(min_in)
        pix_255 = 1. - 1./max_bit
        log_probs = torch.where(x.repeat(1, 1, self.num_mix, 1, 1) >= pix_255,
                                log_one_minus_cdf_min,
                                log_probs)

        # MB x M x H x W
        log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(self.logit_probs, dim=1)
        # now get rid of the mixtures with log sum exp
        log_probs = torch.logsumexp(log_probs, 1)  # MB x H x W
        log_probs = log_probs.sum(dim=[1, 2])  # MB
        return log_probs

    def sample(self, t=None):
        # sample mixture num
        eps = torch.empty_like(self.logit_probs).uniform_(1e-5, 1. - 1e-5)  # MB, M, H, W
        amax = torch.argmax(self.logit_probs - torch.log(-torch.log(eps)), dim=1)
        sel = one_hot(amax, self.logit_probs.size()[1], dim=1, device=self.means.device).unsqueeze(1) # MB, 1, M, H, W

        # select logistic parameters -> MB, 3, H, W
        means = (self.means * sel).sum(2)
        log_scales = (self.log_var * sel).sum(2)
        coefs = (self.coeffs * sel).sum(2)
        if t is not None:
            log_scales = log_scales + torch.ones_like(self.log_scales) * math.log(t)

        # sample from logistic & clip to interval
        u = torch.empty_like(means).uniform_(1e-5, 1. - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

        x_0 = torch.clamp(x[:, 0:1, :, :], -1., 1.)
        x_1 = torch.clamp(x[:, 1:2, :, :] + coefs[:, 0:1, :, :] * x_0, -1., 1.)
        x_2 = torch.clamp(x[:, 2:3, :, :] + coefs[:, 1:2, :, :] * x_0 + coefs[:, 2:3, :, :] * x_1, -1., 1.)
        return torch.cat([x_0, x_1, x_2], dim=1)

    def get_E(self):
        raise NotImplementedError


class MixtureOfGaussians(Distribution):
    def __init__(self, means, logvars, num_comp):
        super().__init__()
        # MB, num_comp, Ch, H, W = means.shape
        self.means = means # MB x num_comp x dims
        self.log_var = logvars  # MB x num_comp x dims
        assert num_comp == self.means.shape[1], 'Number of components should be the same as the number of means'
        self.num_components = num_comp
        self.const = nn.Parameter(torch.log(torch.tensor(self.num_components).float()), requires_grad=False)

    def log_prob(self, x, reduce_dim=True):
        MB = x.shape[0]
        # compute log probs for each component
        x = x.unsqueeze(1)
        log_p_i = -0.5 * (math.log(2.0*math.pi) +
                        self.log_var +
                        torch.pow(x - self.means, 2) / (torch.exp(self.log_var) + 1e-10))
        log_p_i = log_p_i.reshape(MB, self.num_components, -1)
        log_p = torch.logsumexp(log_p_i, dim=1) - self.const
        if reduce_dim:
            return log_p.sum(1)
        return log_p

    def sample(self, N=None, t=None):
        # if N is not given sample one
        if N is None:
            N = 1
            
        comp_idx = torch.randint(0, self.num_components, (N,))
        
        samples = Normal(self.means[:, comp_idx], self.log_var[:, comp_idx]).sample(t=t).squeeze(1).squeeze(0)
        
        return samples


class DiscretizedGaussian(Distribution):
    def __init__(self, mean, logvar):
        super().__init__()
        self.mean = mean
        self.log_var = logvar
        self.inv_std = torch.exp(-logvar)
        self.num_vals = 2**8

    def log_prob(self, x):
        assert (
            x.min() >= -1.0 and (x.min() < 0) and x.max() <= 1.0
        ), "x should be in [-1, 1]"
        # convert to integers [0, 255]
        x = ((x + 1.0) / 2.0 * (self.num_vals - 1)).round().long()
        x_onehot = torch.nn.functional.one_hot(x, num_classes=self.num_vals)

        # create tensor of all possible discrete values
        x_vals = torch.arange(0, self.num_vals)[:, None].repeat(
            1, x.shape[1]
        )  # (256, 3)
        x_vals = (2 * x_vals + 1) / self.num_vals - 1  #  mapped to [-1, 1]
        x_vals = x_vals.transpose(1, 0)[None, None, None, :, :]  # (1, 1, 1, 3, 256)
        x_vals = x_vals.swapaxes(2, 3).swapaxes(1, 2).to(x.device)  # (1, 3, 1, 1, 256)

        # Compare predicted and actual
        centered_x_vals = self.mean[..., None] - x_vals  # (MB, ch, h, w, 256)
        # calculate CDF (k + 1/255)
        plus_in = self.inv_std[..., None] * (
            centered_x_vals + 1.0 / (self.num_vals - 1)
        )
        cdf_plus = approx_standard_normal_cdf(plus_in)

        # calculate CDF (k - 1/255)

        min_in = self.inv_std[..., None] * (centered_x_vals - 1.0 / (self.num_vals - 1))
        cdf_min = approx_standard_normal_cdf(min_in)
        logprobs = torch.log((cdf_plus - cdf_min).clamp(min=1e-10))
        # sum logprobs for the
        loglik = torch.sum(x_onehot * logprobs, axis=-1).reshape(x.shape[0], -1).sum(1)
        return loglik

    def sample(self, t=None):
        # Fake sample : pretend it is not discretized
        if t is not None:
            std = (1 / self.inv_std) * t
            # torch.exp(
            # 0.5 * (self.inv_std + torch.ones_like(self.log_var) * math.log(t))
            # )
        else:
            std = 1 / self.inv_std
        x = self.mean + std * torch.randn_like(self.mean)
        return x

    def get_E(self):
        return self.mean
    