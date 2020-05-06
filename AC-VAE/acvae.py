import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_kernel, n_chan, n_latent):
        super(Encoder, self).__init__()
        self.n_kernel = n_kernel
        self.n_chan = n_chan
        self.n_latent = n_latent
        self.encoder = nn.Sequential(
            # in_ch, out_ch, kernel_size
            nn.Conv2d(1, self.n_kernel, (1, 25), padding=(0, 12), bias=False),
            # num_channels
            nn.BatchNorm2d(self.n_kernel, eps=1e-05, momentum=0.1),
            nn.ELU(),

            nn.Conv2d(self.n_kernel, self.n_kernel, (self.n_chan, 1),
                      padding=(0, 0), stride=1, bias=False),
            nn.BatchNorm2d(self.n_kernel, eps=1e-05, momentum=0.1),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 2)),  # downsamples the signal
            nn.Flatten(),  # [batch_size, n_kernel*n_samples/2]
        )

    def forward(self, x):
        # x.shape = [batch_size, channels, height, width]
        x = self.encoder(x)
        # nn.Linear takes in_features, out_features
        z_mu = nn.Linear(x.shape[-1], self.n_latent, bias=False)(x)
        z_log_var = nn.Linear(x.shape[-1], self.n_latent, bias=False)(x)
        # [batch_size, n_latent], [batch_size, n_latent]
        return z_mu, z_log_var


class Decoder(nn.Module):
    def __init__(self, n_kernel, n_chan, n_sample, n_nuisance, n_latent):
        super(Decoder, self).__init__()
        self.n_kernel = n_kernel
        self.n_chan = n_chan
        self.n_sample = n_sample
        self.input_dim = n_nuisance + n_latent
        self.dense = nn.Sequential(
            nn.Linear(self.input_dim, int(self.n_sample // 2 * self.n_kernel),
                      bias=False),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d takes: in_ch, out_ch, kernel_size
            nn.ConvTranspose2d(self.n_kernel, self.n_kernel, (self.n_chan, 1),
                               bias=False),
            nn.BatchNorm2d(self.n_kernel, eps=1e-05, momentum=0.1),
            nn.ELU(),

            nn.ConvTranspose2d(self.n_kernel, 1, (1, 25), padding=(0, 12),
                               bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1),
        )

    def forward(self, zs):
        # zs: [batch_size, n_latent+n_nuisance]
        zs = self.dense(zs)
        zs = zs.view(zs.shape[0], 1, zs.shape[-1])
        zs = nn.Upsample(scale_factor=2, mode='nearest')(zs)
        zs = zs.view(zs.shape[0], self.n_kernel, 1, int(zs.shape[-1]/self.n_kernel))
        x_hat = F.sigmoid(self.decoder(zs))
        return x_hat


class Adversary(nn.Module):
    def __init__(self, n_latent, n_nuisance):
        super(Adversary, self).__init__()
        self.n_latent = n_latent
        self.n_nuisance = n_nuisance
        self.adversary = nn.Sequential(
            nn.Linear(n_latent, self.n_nuisance, bias=False),
        )

    def forward(self, z):
        # z: [batch_size, n_latent]
        s_hat = self.adversary(z)
        s_hat = F.softmax(s_hat, dim=1)
        return s_hat


class AdversarialcVAE(nn.Module):
    def __init__(self, adversarial, n_sample, n_nuisance, n_chan=16,
                 n_latent=100, n_kernel=40, lam=0.01):
        super(AdversarialcVAE, self).__init__()
        # Input, data set and model training scheme parameters
        self.n_chan = n_chan
        self.n_sample = n_sample
        self.n_latent = n_latent
        self.n_nuisance = n_nuisance
        self.n_kernel = n_kernel
        self.adversarial = adversarial
        self.lam = lam

        # Build the network blocks
        self.encoder_net = Encoder(self.n_kernel, self.n_chan, self.n_latent)
        self.decoder_net = Decoder(self.n_kernel, self.n_chan, self.n_sample,
                                   self.n_nuisance, self.n_latent)
        self.adversary_net = Adversary(self.n_latent, self.n_nuisance)

    def forward(self, x, y):
        '''
        x is a tuple of torch tensors
        x: (torch.tensor [batch_size, 1, n_chan, n_sample]), input data
        y: (torch.tensor [batch_size, n_nuisance]), nuisance parameters
        '''
        # Encode
        z_mu, z_log_var = self.encoder_net(x)

        # Sample from the distribution having latent parameters z_mu, z_log_var
        # Reparameterize
        std = torch.exp(z_log_var / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(z_mu)  # [batch_size, n_latent]
        zs = torch.cat((z, y), dim=1)  # [batch_size, n_latent+n_nuisance]

        if self.adversarial:
            x_hat = self.decoder_net(zs)
            s_hat = self.adversary_net(z)
            return z_mu, z_log_var, x_hat, s_hat
        else:
            x_hat = self.decoder_net(zs)
            return z_mu, z_log_var, x_hat


if __name__ == "__main__":
    '''
    Unit test to check the network forward propagates without an error, 
    using dummy data x
    '''
    n_sample = 1000
    n_nuisance = 10
    n_latent = 100
    n_chan = 16
    n_kernel = 40
    lam = 0.01
    batch_size = 10
    adversarial = True
    model = AdversarialcVAE(adversarial, n_sample, n_nuisance, n_chan, n_latent,
                            n_kernel, lam)
    input = torch.randn(batch_size, 1, n_chan, n_sample)
    nuisance_par = torch.randn(batch_size, n_nuisance)
    if adversarial:
        z_mu, z_log_var, x_hat, s_hat = model(input, nuisance_par)
        print(z_mu, z_log_var, x_hat, s_hat)
        print(z_mu.shape, z_log_var.shape, x_hat.shape, s_hat.shape)
    else:
        z_mu, z_log_var, x_hat = model(input, nuisance_par)
        print(z_mu, z_log_var, x_hat)
        print(z_mu.shape, z_log_var.shape, x_hat.shape)



