import numpy as np
import pandas as pd
import glob
from scipy.io import loadmat
import concurrent.futures  # for parallel processing
from tqdm import tqdm
import pickle
import argparse


# Global parameters and variables of the dataset
global t0, fs, dt, num_eeg_ch, num_eeg_ch, num_emg_ch, num_force_ch, \
       num_ch, num_conds, data_files, num_subjects, \
       columns, PIK1, PIK2, PIK3
t0 = 0
fs = 1200
dt = 1.0/fs
num_eeg_ch = 14
num_emg_ch = 4
num_force_ch = 3
num_ch = 21
num_conds = 18

# Get a list of files to process
data_files = list(map(loadmat, glob.glob("../../Datasets/HAPTIXData/1200/*.mat")))
num_subjects = len(data_files)
columns = ['EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-6',
           'EEG-7', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12',
           'EEG-13', 'EEG-14', 'EMG-1', 'EMG-2', 'EMG-3', 'EMG-4',
           'Force-x', 'Force-y', 'Force-z']

PIK1 = 'data.dat'
PIK2 = 'movement_labels.dat'
PIK3 = 'surface_labels.dat'

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=400)
args = parser.parse_args()


def extract_data(file):
    '''
    param: file(str), is .mat data file of 1 subject inside the Dataset dir.
    return: data_all_cond(np.array, #trials * #channels * #samples), is the
            preprocessed data of 1 subject
    return: movement_labels(np.array, #trials), is the movement type at each
            trial. rub: 0, and tap: 1
    return: surface_labels(np.array, #trials), is the surface touched at each
            trial. flat: 0, medium-rough: 1, and rough: 2
    '''
    data_all_cond = None
    movement_labels = []
    surface_labels = []
    for cond in tqdm(range(num_conds), ascii=True):
        num_trials = file['EEGSeg_Ch'][0, 0][0, cond].shape[0]
        data_per_cond = np.zeros((num_trials, num_eeg_ch, args.num_samples))
        # if a trial has long enough samples
        if file['EEGSeg_Ch'][0, 0][0, cond].shape[1] >= args.num_samples:
            for trial in range(num_trials):
                movement_labels.append(0) if cond < 9 else movement_labels.append(1)
                if cond % 3 == 0:
                    surface_labels.append(0)
                elif cond % 3 == 1:
                    surface_labels.append(1)
                else:
                    surface_labels.append(2)
                data = pd.DataFrame(columns=columns)
                for ch in range(num_eeg_ch):
                    data.iloc[:, ch] = file['EEGSeg_Ch'][0, ch][0, cond][trial, :]
                for ch in range(num_emg_ch):
                    data.iloc[:, ch + num_eeg_ch] = file['EMGSeg_Ch'][0, ch][
                                                         0, cond][trial, :]
                for ch in range(num_force_ch):
                    data.iloc[:, ch + num_eeg_ch + num_emg_ch] = \
                    file['ForceSeg_Ch'][0, ch][0, cond][trial, :]

                # High pass filter by mean subtraction in each trial
                # from the eeg and emg columns in order to remove dc drift
                idx = num_eeg_ch + num_emg_ch
                data.iloc[:, :idx] -= data.iloc[:, :idx].mean()
                # convert volts to microvolts for EEG and EMG channels
                data.iloc[:, :idx] *= 1e6
                # convert volts to milivolts for force channels
                data.iloc[:, idx:] *= 1e3
                # use all channels of EEG, EMG and Force
                # and up to num_samples number of samples
                data = data.iloc[:args.num_samples, :]
                data_per_cond[trial] = data.to_numpy().T[:num_eeg_ch, :]
            if data_all_cond is None:
                data_all_cond = data_per_cond
            else:
                data_all_cond = np.concatenate((data_all_cond, data_per_cond), axis=0)
    movement_labels = np.asarray(movement_labels)
    surface_labels = np.asarray(surface_labels)
    return data_all_cond, movement_labels, surface_labels


def save_data(data, movement_labels, surface_labels):
    with open(PIK1, "wb") as f:
        pickle.dump(data, f)
    with open(PIK2, "wb") as f:
        pickle.dump(movement_labels, f)
    with open(PIK3, "wb") as f:
        pickle.dump(surface_labels, f)


if __name__ == "__main__":
    # Reading and processing in parallel.')
    # Create a pool of processes.
    # By default, one is created for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Process the list of files, but split the work across the process pool
        # to use all CPUs.
        # executor.map() function takes in the helper function to call and
        # the list of data to process with it.
        # executor.map() function returns results in  the same order as the
        # list of data given to the process.
        data, movement_labels, surface_labels = [], [], []
        for subject in tqdm(executor.map(extract_data, data_files),
                            total=len(data_files)):
            data.append(subject[0])
            movement_labels.append(subject[1])
            surface_labels.append(subject[2])

    save_data(data, movement_labels, surface_labels)
    print('pre-processing finished successfully.')