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
CONVERSION_RATIO = np.sqrt(1/6.5) #conversion ratio to be used for conveting the animal data to compare it

def get_frequency_and_amplitude(time, signal):
    n = len(time)
    dt = time[1] - time[0]

    freqs = np.fft.rfftfreq(n, dt)
    fft_values = np.fft.rfft(signal - np.mean(signal))

    amplitude_spectrum = np.abs(fft_values) / n
    idx = np.argmax(amplitude_spectrum[1:]) + 1

    freq = freqs[idx]
    amplitude = 2 * amplitude_spectrum[idx]   # keep same unit as signal (rad here)
    phase = np.angle(fft_values[idx])         # radians

    return freq, amplitude, phase

def get_animal_data(path):

    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    freq = np.zeros(8)
    amp = np.zeros(8)
    ipls = np.zeros(7)

    times = data[10:-10, 0]
    joint_angles = np.deg2rad(data[10:-10, 1:9])  # convert animal angles to rad

    n_joints = 8

    for i in range(n_joints):
        freq[i], amp[i], _ = get_frequency_and_amplitude(times, joint_angles[:, i])

    for i in range(7):
        _, _, phase_i = get_frequency_and_amplitude(times, joint_angles[:, i])
        _, _, phase_i_plus_1 = get_frequency_and_amplitude(times, joint_angles[:, i + 1])
        ipls[i] = (phase_i_plus_1 - phase_i) % (2 * np.pi)

    return freq, amp, ipls

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
    #pylog.warning("TODO: 2.3 Analyze the provided animal data and compare the animal locomotion performance with your implemented controller.")
    # pylog.set_level('critical')

    results_dir = os.path.join(BASE_PATH, PLOT_PATH)
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

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
    freq_sim, amp_sim, ipls_sim = get_sim_data(sim_result)

    data = np.genfromtxt(ANIMAL_DATA_PATH, delimiter=',', skip_header=1)
    time_animal = data[:, 0]
    joints_animal = np.deg2rad(data[:, 1:])  # convert to rad for consistent plotting

    with open(ANIMAL_DATA_PATH, 'r') as f:
        header = f.readline().strip().split(',')
        joint_names_animal = header[1:] 

    # Synchroniser les longueurs
    min_len = min(len(sim_times), len(time_animal), sensor_data_joints_positions.shape[0])
    joints_actifs = sensor_data_joints_positions[:min_len, indices_actifs]  # already in rad

    # Start simulation after 0.2 sec 
    start_time = 0.2  # seconds
    mask = time_animal >= start_time
    time_cut = time_animal[mask]
    time_cut = time_cut - time_cut[0]
    joints_actifs_cut = joints_actifs[mask]

    # ---- Pas besoin de synchroniser les longueurs, on trace chacun sur son propre temps ----
    
    # set figure size to width:height = 5:2
    fig, axs = plt.subplots(1,2, figsize=(10, 4))

    # L'animal a été filmé sur environ 0.75s
    animal_duration = 0.75        
    t_animal_start = 0.0   

    # La Magie de Froude : T_robot = T_animal / CONVERSION_RATIO
    # On calcule la durée équivalente pour le robot géant
    scaled_duration = animal_duration / CONVERSION_RATIO
    
    t_sim_start = 2.375   # On skip le régime transitoire de la simulation

    # Masques pour isoler les bonnes fenêtres temporelles respectives
    mask_sim = (sim_times >= t_sim_start) & (sim_times <= t_sim_start + scaled_duration)
    mask_anim = (time_animal >= t_animal_start) & (time_animal <= t_animal_start + animal_duration)

    # Création des axes de temps relatifs
    t_sim_plot = sim_times[mask_sim] - t_sim_start
    t_anim_plot = time_animal[mask_anim] - t_animal_start
    
    # ÉTAPE CLÉ : On étire le temps de l'animal pour qu'il corresponde au temps du robot
    t_anim_plot_scaled = t_anim_plot / CONVERSION_RATIO

    # 1. Simulation Joints (plot in radians)
    for i, idx in enumerate(indices_actifs):
        axs[0].plot(t_sim_plot, sensor_data_joints_positions[mask_sim, idx], label=noms_actifs[i])
    axs[0].set_title('Simulation - Active Joints (Steady State)')
    axs[0].set_ylabel('Angle [rad]')
    axs[0].set_xlim(0, scaled_duration) 
    axs[0].legend(loc='upper right', fontsize=8, ncol=2)
    axs[0].grid(True)

    # 2. Animal Joints (Time-scaled, already in radians)
    for i in range(joints_animal.shape[1]):
        axs[1].plot(t_anim_plot_scaled, joints_animal[mask_anim, i], label=joint_names_animal[i])
    axs[1].set_title('Animal Joints (Time Scaled by Froude Ratio)')
    axs[1].set_xlabel('Scaled Relative Time [s] (Equivalent Robot Time)')
    axs[1].set_ylabel('Angle [rad]')
    axs[1].set_xlim(0, scaled_duration)
    axs[1].legend(loc='upper right', fontsize=8, ncol=2)
    axs[1].grid(True)

    plt.tight_layout()

    fig.savefig(os.path.join(BASE_PATH, PLOT_PATH, "sim_vs_animal_joint_angles_2_3.png"), dpi=150)
    # plt.show()  # remove unconditional show

    # Fréquence
    freq_animal_scaled = freq_animal * CONVERSION_RATIO
    

    freq_error = np.mean((freq_sim - freq_animal_scaled)**2)**0.5

    # Amplitude
    amp_error = np.mean((amp_sim - amp_animal)**2)**0.5

    # IPLs
    ipl_error = np.mean((ipls_sim - ipls_animal)**2)**0.5

    print("=== Animal Data (Raw) ===")
    print(f"Frequencies Animal: {np.mean(freq_animal):.3f} Hz")
    print(f"Amplitudes (r) Animal: {np.mean(amp_animal):.3f} rad")
    print(f"IPLs Animal : {np.mean(ipls_animal):.3f} rad\n")

    print("=== Simulation Data ===")
    print(f"Simulation Frequency: {np.mean(freq_sim):.3f} Hz")
    print(f"Simulation Amplitude (r): {np.mean(amp_sim):.3f} rad")
    print(f"Simulation IPLs: {np.mean(ipls_sim):.3f} rad\n")

    print("=== Errors (with Froude Scaling for Frequency) ===")
    # On calcule la fréquence scalée juste pour l'affichage
    freq_animal_scaled = freq_animal * CONVERSION_RATIO
    print(f"(Note: Animal frequency scaled for robot size is {np.mean(freq_animal_scaled):.3f} Hz)")
    print(f"Frequency Error: {freq_error:.3f} Hz")
    print(f"Amplitude Error: {amp_error:.3f} rad")
    print(f"IPLs Error: {ipl_error:.3f} rad")
    '''
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

    # Amplitude Envelope
    axs2[0].plot(range(8), amp_sim, marker='o', label='Simulation', linewidth=2)
    axs2[0].plot(range(8), amp_animal, marker='s', label='Animal', linewidth=2)
    axs2[0].set_title('Amplitude Envelope')
    axs2[0].set_xlabel('Joint Index (0=Head, 7=Tail) [-]')
    axs2[0].set_ylabel('Amplitude [rad]')
    axs2[0].grid(True)
    axs2[0].legend()

    # Local IPLs
    axs2[1].plot(range(7), ipls_sim, marker='o', label='Simulation', linewidth=2)
    axs2[1].plot(range(7), ipls_animal, marker='s', label='Animal', linewidth=2)
    axs2[1].set_title('Intersegmental Phase Lags (IPL)')
    axs2[1].set_xlabel('Joint Pair Index [-]')
    axs2[1].set_ylabel('Phase Lag [rad]')
    axs2[1].grid(True)
    axs2[1].legend()

    fig2.tight_layout()
    fig2.savefig(os.path.join(BASE_PATH, PLOT_PATH, "amplitude_ipl_comparison_2_3.png"), dpi=150)'''

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_3(plot=True)

