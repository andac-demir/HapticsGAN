import torch
import torch.optim as optim
import numpy as np
from acvae import AdversarialcVAE
from train_acvae import train, test
import argparse
import pickle
from math import floor
from keras.utils import to_categorical
from tqdm import tqdm


PIK1 = 'data.dat'
PIK2 = 'movement_labels.dat'
PIK3 = 'surface_labels.dat'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=1.)
    parser.add_argument("--n_nuisance", type=int, default=2)
    parser.add_argument("--n_latent", type=int, default=1000)
    parser.add_argument("--n_kernel", type=int, default=80)
    parser.add_argument("--adversarial", type=str2bool, default='true')
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_chan", type=int, default=14)
    parser.add_argument("--n_sample", type=int, default=400)
    parser.add_argument("--inference", type=str2bool, default='false')
    args = parser.parse_args()
    return args


def load_data(PIK1, PIK2, PIK3):
    with open(PIK1, "rb") as f:
        data = pickle.load(f)
    with open(PIK2, "rb") as f:
        movement_labels = pickle.load(f)
    with open(PIK3, "rb") as f:
        surface_labels = pickle.load(f)
    return data, movement_labels, surface_labels


def get_batches(data, movement_labels, surface_labels, batch_size, n_chan, n_sample):
    ratio = 0.70  # use 70 percent of the data for training and rest for testing
    train_iterator, test_iterator = [], []

    # concatenate a list of arrays into a single array
    data = np.concatenate(data)  # [total_trials, num_channels, num_samples]
    movement_labels = np.concatenate(movement_labels)  # [total_trials]
    surface_labels = np.concatenate(surface_labels)  # [total_trials]

    # shuffle data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = torch.from_numpy(data[indices])
    movement_labels = movement_labels[indices]
    surface_labels = surface_labels[indices]

    # get one hot encodings of movement and surface labels:
    movement_labels = torch.from_numpy(to_categorical(movement_labels))
    surface_labels = torch.from_numpy(to_categorical(surface_labels))

    # split data into batches
    n_batches = floor(data.shape[0]/batch_size)
    end = n_batches * batch_size
    data = torch.split(data[:end], batch_size)
    movement_labels = torch.split(movement_labels[:end], batch_size)
    surface_labels = torch.split(surface_labels[:end], batch_size)

    # get the train and test data
    end = int(ratio * n_batches)
    train_surf_labels, test_surf_labels = surface_labels[:end], surface_labels[end:]
    # create data iterators
    for i in range(end):
        # reshape the batch of train data
        x = data[i].view(batch_size, 1, n_chan, n_sample).float()
        y = movement_labels[i].float()
        train_iterator.append((x, y))
    for i in range(end, n_batches):
        # reshape the batch of train data
        x = data[i].view(batch_size, 1, n_chan, n_sample).float()
        y = movement_labels[i].float()
        test_iterator.append((x, y))
    return train_iterator, test_iterator, train_surf_labels, test_surf_labels


def main():
    args = parseArgs()
    data, movement_labels, surface_labels = load_data(PIK1, PIK2, PIK3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdversarialcVAE(args.adversarial, args.n_sample, args.n_nuisance,
                            args.n_chan, args.n_latent, args.n_kernel, args.lam)
    # weight decay is L2 regularization in parameter updates
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    train_iterator, test_iterator, train_surf_labels, test_surf_labels = \
        get_batches(data, movement_labels, surface_labels, args.batch_size,
                    args.n_chan, args.n_sample)
    print("\n\nData parsed.")
    train(model, device, optimizer, args.adversarial, train_iterator,
          test_iterator, args.batch_size, args.n_epoch)
    print('Finished Training')
    # Save our trained model
    PATH = 'saved_model/inference_net.pth'
    torch.save(model.state_dict(), PATH)
    if args.adversarial:
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(test_iterator):
                test_data, test_nuisance = batch
                outputs = model(test_data, test_nuisance)
                s_hat = outputs[3]
                _, predicted = torch.max(s_hat.data, 1)
                _, actual = torch.max(test_nuisance.data, 1)
                total += test_nuisance.size(0)
                correct += (predicted == actual).sum().item()
        print('Accuracy of the network on the test dataset: %d %%' % (
                100 * correct / total))

if __name__ == "__main__":
    main()