#!/usr/bin/env python3

import os
import h5py
import matplotlib.pyplot as plt

import numpy as np

from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import runsim


BASE_PATH = 'logs/exercise2_3/'
PLOT_PATH = 'results'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'

def get_frequency_and_amplitude(time, signal):
    n = len(time)
    dt = time[1] - time[0]

    freqs = np.fft.rfftfreq(n, dt)
    fft_values = np.fft.rfft(signal - np.mean(signal))

    amplitude_spectrum = np.abs(fft_values) / n

    idx = np.argmax(amplitude_spectrum[1:]) + 1

    freq = freqs[idx]
    amplitude = 2 * amplitude_spectrum[idx] * 180 / np.pi # Convertir en degrés
    phase = np.angle(fft_values[idx])  * 180 / np.pi

    return freq, amplitude, phase

def get_animal_data(path):

    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    freq = np.zeros(8)
    amp = np.zeros(8)
    ipls = np.zeros(7)


    times = data[10:-10, 0]
    joint_angles = data[10:-10, 1:9]

    n_joints = 8

    for i in range(n_joints):
        freq[i], amp[i], _ = get_frequency_and_amplitude(times, joint_angles[:, i])

    for i in range(7):
        _, _, phase_i = get_frequency_and_amplitude(times, joint_angles[:, i])
        _, _, phase_i_plus_1 = get_frequency_and_amplitude(times, joint_angles[:, i + 1])
        ipls[i] = (phase_i_plus_1 - phase_i) % (2 * np.pi)

    return freq, np.deg2rad(amp), ipls

def get_sim_data(path):

    with h5py.File(path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    n_joints = 8

    freq = np.zeros(n_joints)
    amp = np.zeros(n_joints)
    ipls = np.zeros(n_joints - 1)

    for i in range(n_joints):
        freq[i], amp[i], _ = get_frequency_and_amplitude(sim_times, sensor_data_joints_positions[:, i])

    for i in range(n_joints - 1):
        _, _, phase_i = get_frequency_and_amplitude(sim_times, sensor_data_joints_positions[:, i])
        _, _, phase_i_plus_1 = get_frequency_and_amplitude(sim_times, sensor_data_joints_positions[:, i + 1])
        ipls[i] = (phase_i_plus_1 - phase_i) % (2 * np.pi)

    return freq, amp, ipls


def exercise2_3(**kwargs):
    """ex2.3 main"""
    pylog.warning("TODO: 2.3 Analyze the provided animal data and compare the animal locomotion performance with your implemented controller.")
    # pylog.set_level('critical')

    # Load HDF5 from 2_1
    sim_result = 'logs/exercise2_1/' + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
        joints_names = f['FARMSLISTanimats']['0']['sensors']['joints']['names'][:]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    #### Analyze simulation data ####
    joints_names_decoded = [name.decode('utf-8') for name in joints_names]
    indices_actifs = list(range(8))
    noms_actifs = [joints_names_decoded[i] for i in indices_actifs]

    #### Analyze animal data ####
    freq_animal, amp_animal, ipls_animal = get_animal_data(ANIMAL_DATA_PATH)
    print("Frequencies Animal:", np.mean(freq_animal))
    print("Amplitudes (r) Animal:", np.mean(amp_animal))
    print("IPLs Animal :", np.mean(ipls_animal))

    ### Analyze simulation data ###
    freq_sim, amp_sim, ipls_sim = get_sim_data(sim_result)
    print("Simulation Frequency:", np.mean(freq_sim))
    print("Simulation Amplitude (r):", np.mean(amp_sim))
    print("Simulation IPLs:", np.mean(ipls_sim))

    data = np.genfromtxt(ANIMAL_DATA_PATH, delimiter=',', skip_header=1)
    time_animal = data[:, 0]
    joints_animal = data[:, 1:]

    with open(ANIMAL_DATA_PATH, 'r') as f:
        header = f.readline().strip().split(',')
        joint_names_animal = header[1:] 

    # Synchroniser les longueurs
    min_len = min(len(sim_times), len(time_animal), sensor_data_joints_positions.shape[0])
    joints_actifs = sensor_data_joints_positions[:min_len, indices_actifs]
    joints_actifs = (joints_actifs * 180 / np.pi ) * (1/np.sqrt(6.5)) # Ratio f_simulation/f_animal  = sqrt(6.5)


    # Start simulation after 0.2 sec 
    start_time = 0.2  # seconds
    mask = time_animal >= start_time
    time_cut = time_animal[mask]
    time_cut = time_cut - time_cut[0]
    joints_actifs_cut = joints_actifs[mask]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Simulation Joints 
    for i, idx in enumerate(indices_actifs):
        axs[0].plot(time_cut, joints_actifs_cut[:, i], label=noms_actifs[i])
    axs[0].set_title('Simulation - Active Joints')
    axs[0].set_ylabel('Angle [deg]')
    axs[0].legend()
    axs[0].grid(True)

    # Animal Joints
    for i in range(joints_animal.shape[1]):
        axs[1].plot(time_animal[:min_len], joints_animal[:min_len, i], label=joint_names_animal[i])
    axs[1].set_title('Animal Joints')
    axs[1].set_ylabel('Angle [deg]')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Fréquence
    freq_error = np.mean((freq_sim - freq_animal)**2)**0.5

    # Amplitude
    amp_error = np.mean((amp_sim - amp_animal)**2)**0.5

    # IPLs
    ipl_error = np.mean((ipls_sim - ipls_animal)**2)**0.5

    print(f"Frequency Error: {freq_error:.3f} Hz")
    print(f"Amplitude Error: {amp_error:.3f} deg")
    print(f"IPLs Error: {ipl_error:.3f} deg")

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_3(plot=True)

