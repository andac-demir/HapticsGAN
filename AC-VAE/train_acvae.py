import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from acvae import AdversarialcVAE


def vae_loss(x, x_hat, z_mu, z_log_var):
    '''
    loss = reconstruction_loss + kl_divergence
    '''
    # reconstruction loss
    RCL = F.binary_cross_entropy(x_hat, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
    return RCL + KLD


def adversary_loss(lam, s, s_hat, z_mu, z_log_var):
    '''
    loss = adversary_loss + kl_divergence
    '''
    # categorization loss
    CL = lam * nn.BCELoss()(s_hat, s)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
    return CL + KLD


def train_epoch(model, device, optimizer, train_iterator, adversarial):
    '''
    trains one epoch.
    if training adversarial conditional VAE, alternate between training VAE and
    adversary network
    '''
    model.train()
    train_cvae_loss = 0  # epoch loss for cvae, adversarial is True
    train_adversarial_loss = 0  # epoch loss for the adversarial net
    train_loss = 0  # epoch loss, adversarial is False
    for i, (x, y) in enumerate(train_iterator):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # update the gradients to zero

        if adversarial:
            # first forward pass
            z_mu, z_log_var, x_hat, s_hat = model(x, y)
            # calculate adversary loss
            loss = adversary_loss(model.lam, y, s_hat, z_mu, z_log_var)
            # backward pass (freezing decoder)
            trainable_params = []
            for name, p in model.named_parameters():
                if "decoder_net" in name:
                    p.requires_grad = False
                else:
                    if "adversary_net" in name:
                        p.requires_grad = True
                    trainable_params.append(p)
            optimizer = optim.Adam(params=trainable_params, lr=1e-3,
                                   weight_decay=1e-2)
            # update the parameters of encoder and adversarial network
            loss.backward()
            train_adversarial_loss += loss.item()
            optimizer.step()  # update the weights

            # second forward pass (freezing the adversary)
            z_mu, z_log_var, x_hat, s_hat = model(x, y)
            # calculate vae loss
            loss = vae_loss(x, x_hat, z_mu, z_log_var)
            # backward pass (freezing the adversary)
            trainable_params = []
            for name, p in model.named_parameters():
                if "adversary_net" in name:
                    p.requires_grad = False
                else:
                    if "decoder_net" in name:
                        p.requires_grad = True
                    trainable_params.append(p)
            optimizer = optim.Adam(params=trainable_params, lr=1e-3,
                                   weight_decay=1e-2)
            # update the parameters of encoder and decoder
            loss.backward()
            train_cvae_loss += loss.item()
            optimizer.step()  # update the weights
            return train_adversarial_loss, train_cvae_loss

        else:
            # forward pass
            z_mu, z_log_var, x_hat = model(x, y)
            loss = vae_loss(x, x_hat, z_mu, z_log_var)
            # backward pass
            loss.backward()
            train_loss += loss.item()
            optimizer.step()  # update the weights
            return train_loss


def train(model, device, optimizer, adversarial,
          train_iterator, test_iterator, batch_size, n_epoch=100):
    for e in range(n_epoch):
        if adversarial:
            train_adversarial_loss, train_cvae_loss = train_epoch(model, device,
                                        optimizer, train_iterator, adversarial)
            test_adversarial_loss, test_cvae_loss = test(model, device,
                                                         test_iterator, adversarial)
            train_adversarial_loss /= (len(train_iterator) * batch_size)
            train_cvae_loss /= (len(train_iterator) * batch_size)
            test_cvae_loss /= (len(test_iterator) * batch_size)
            test_adversarial_loss /= (len(test_iterator) * batch_size)
            print(f'Epoch {e}, Train Adversary Loss: {train_adversarial_loss:.4f}, '
                  f'Train CVAE Loss: {train_cvae_loss:.4f}, '
                  f'Test Adversary Loss: {test_adversarial_loss:.4f}, '
                  f'Test CVAE Loss: {test_cvae_loss:.4f}')
        else:
            train_loss = train_epoch(model, device, optimizer,
                                     train_iterator, adversarial)
            test_loss = test(model, device, test_iterator, adversarial)
            train_loss /= (len(train_iterator) * batch_size)
            test_loss /= (len(test_iterator) * batch_size)
            print(f'Epoch {e}, Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}')


def test(model, device, test_iterator, adversarial):
    # set the evaluation mode
    model.eval()
    # test loss for the data
    test_adversarial_loss = 0
    test_cvae_loss = 0
    test_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            x, y = x.to(device), y.to(device)
            # forward pass
            if adversarial:
                z_mu, z_log_var, x_hat, s_hat = model(x, y)
                test_adversarial_loss += adversary_loss(model.lam, y, s_hat,
                                                        z_mu, z_log_var)
                test_cvae_loss += vae_loss(x, x_hat, z_mu, z_log_var).item()
                return test_adversarial_loss, test_cvae_loss
            else:
                z_mu, z_log_var, x_hat = model(x, y)
                test_loss = vae_loss(x, x_hat, z_mu, z_log_var).item()
                return test_loss


# Unit test on the dummy data
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epoch = 100
    adversarial = True
    n_nuisance = 2
    n_chan = 14
    n_sample = 100
    model = AdversarialcVAE(adversarial, n_sample, n_nuisance,
                            n_chan, n_latent=100, n_kernel=80, lam=1.)
    # weight decay is L2 regularization in parameter updates
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # create a dummy dataset
    train_ratio, test_ratio = 70, 30
    batch_size = 32
    train_iterator, test_iterator = [], []
    for i in range(train_ratio):
        x = torch.randn(batch_size, 1, n_chan, n_sample)
        y = torch.randn(batch_size, n_nuisance)
        train_iterator.append((x, y))
    for i in range(test_ratio):
        x = torch.randn(batch_size, 1, n_chan, n_sample)
        y = torch.randn(batch_size, n_nuisance)
        test_iterator.append((x, y))

    train(model, device, optimizer, adversarial, train_iterator, test_iterator,
          batch_size, n_epoch)