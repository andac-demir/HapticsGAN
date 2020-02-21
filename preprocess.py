import numpy as np
import pandas as pd
import glob
from scipy.io import loadmat
import concurrent.futures # for parallel processing
from tqdm import tqdm


# Get the participant data from Dataset directory
global t0, fs, dt, num_eeg_ch, eeg_channels, num_conds
t0 = 0
fs = 1200
dt = 1.0/fs
num_eeg_ch = 3
eeg_channels = [4, 5, 6]
num_conds = 18

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
                    data.iloc[:, i] = file['EEGSeg_Ch'][0, ch][0, cond][trial, :]

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


def main():
    '''
    # both train and test data is a list (each participant)
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


if __name__ == '__main__':
    main()