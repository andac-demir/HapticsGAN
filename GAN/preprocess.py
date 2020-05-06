import pandas as pd
import glob
from scipy.io import loadmat
import concurrent.futures # for parallel processing
from tqdm import tqdm
import pickle
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Get the participant data from Dataset directory
global t0, fs, dt, num_eeg_ch, eeg_channels, num_conds, N_samples, time
t0 = 0
fs = 1200
dt = 1.0/fs
num_eeg_ch = 3
eeg_channels = [4, 5, 6]
num_conds = 18
N_samples = 400
time = np.arange(0, N_samples) * dt

# Get a list of files to process
data_files = list(map(loadmat, glob.glob("Dataset/*.mat")))
num_subjects = len(data_files)
columns = ['EEG-5', 'EEG-6', 'EEG-7']


def extract_data(file):
    data_splitBy_trials = []  # list of data frames, each df corresponding to a trial

    for cond in tqdm(range(num_conds), ascii=True):
        num_trials = file['EEGSeg_Ch'][0, 0][0, cond].shape[0]
        # if a trial has long enough samples
        if file['EEGSeg_Ch'][0, 0][0, cond].shape[1] >= 400:
            for trial in range(num_trials):
                data = pd.DataFrame(columns=columns)
                for i, ch in enumerate(eeg_channels):
                    data.iloc[:, i] = file['EEGSeg_Ch'][0, ch][0, cond][trial, :N_samples]

                # mean subtraction in each trial from the eeg for removing dc drift
                data.iloc[:, :num_conds] -= data.iloc[:, :num_conds].mean()
                # convert volts to microvolts for EEG
                data.iloc[:, :num_conds] *= 1e6
                data_splitBy_trials.append(data)

    train_data, test_data = train_test_split(data_splitBy_trials)
    return train_data, test_data


def train_test_split(data, train_ratio=0.50):
    '''
    with train ratio=x, we use x ratio of the trials from a participant for training
    and other trials for testing
    '''
    num_train = int(train_ratio * len(data))
    train_data, test_data = data[:num_train], data[num_train:]
    return train_data, test_data


def get_spectogram(trial):
    '''
    create time-frequency representation of each trial
    https://www.mathworks.com/help/wavelet/examples/classify-time-series-using-wavelet-analysis-and-deep-learning.html
    We see that a higher scale-factor (longer wavelet) corresponds with a
    smaller frequency, so by scaling the wavelet in the time-domain
    we will analyze smaller frequencies (achieve a higher resolution)
    in the frequency domain.
    And vice versa, by using a smaller scale we have more detail
    in the time-domain.
    '''
    scales = np.arange(1, 128)
    cwtmatr, freqs = pywt.cwt(trial, scales, 'morl', sampling_period=dt)
    # ignore freqs > 100 Hz
    start = len(freqs[freqs > 100])
    freqs = freqs[start:]
    cwtmatr = cwtmatr[start:, :]
    power = (abs(cwtmatr)) ** 2
    levels = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512,
              640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
              2048]
    contour_levels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, freqs, np.log2(power), contour_levels,
                     extend='both', cmap=plt.cm.seismic)
    ax.set_ylabel('Approximate Frequency [Hz]', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_xlim(time.min(), time.max())
    ax.set_title('Wavelet Transform (Power Spectrum)', fontsize=20)
    yticks = np.arange(0, np.round(freqs.max()/100)*100, step=100)
    ax.set_yticks(yticks)
    plt.show()


def main():
    '''
    both train and test data is a list (each participant)
    of list (each trial) of data frames
    '''
    train_data = []
    test_data = []
    print('Reading and processing in parallel %i subjects.' %num_subjects)

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Process the list of files, but split the work across the process pool to use all CPUs.
        # executor.map() function takes in the helper function to call and the list of data to process with it.
        # executor.map() function returns results in  the same order as the list of data given to the process.
        for train, test in tqdm(executor.map(extract_data, data_files),
                                total=len(data_files)):
            train_data.append(train)
            test_data.append(test)

    total_train, total_test = 0, 0
    for train in train_data:
        total_train += len(train)
    for test in test_data:
        total_test += len(test)
    print('Total number of trials of all the conditions and subjects '
          'in the train file: %i' % total_train)
    print('Total number of trials of all the conditions and subjects '
          'in the test file: %i' % total_test)
    # first 10 samples of the first trial from the first subject
    print(train_data[0][0].head(10))

    # save train and test data:
    PIK1 = 'Dataset/train_data.dat'
    PIK2 = 'Dataset/test_data.dat'
    with open(PIK1, "wb") as f:
        pickle.dump(train_data, f)
    with open(PIK2, "wb") as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    main()