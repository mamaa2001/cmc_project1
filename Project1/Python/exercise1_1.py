#!/usr/bin/env python3

import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np

from farms_core.utils.profile import profile
from farms_core import pylog

from simulate import runsim
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_frequency_amplitude_fft,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

BASE_PATH = 'logs/exercise1_1/'
PLOT_PATH = 'results'


def post_processing():
    """Post processing"""
    # Load HDF5
    sim_result = BASE_PATH + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
        joints_names = f['FARMSLISTanimats']['0']['sensors']['joints']['names'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]
    
    # Load Controller
    with open(BASE_PATH + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)

    # Metrics computation
    # pylog.warning("TODO: 1.1: Complete metrics implementation in metrics.py")
    freq, _, amp = compute_frequency_amplitude_fft(
        times=sim_times, smooth_signals=neural_signals_smoothed)

    inds_couples = [[i, i+1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    _, ipls_mean = compute_neural_phase_lags(times=sim_times,
                                             smooth_signals=neural_signals_smoothed,
                                             freqs=freq,
                                             inds_couples=inds_couples)

    mech_freq, mech_amp = compute_mechanical_frequency_amplitude_fft(
        times=sim_times,
        signals=sensor_data_joints_positions[:, :8],
    )

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )

    energy, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    # pylog.warning("TODO: 1.2: Verify the computed metrics are consistent with the expected values")
    print('Estimated neural metrics:')
    print('Frequencies: ', freq, '\nAmplitudes: ', amp,
          '\nMean phase lags (radians): ', ipls_mean)
    print('Estimated mechanical metrics:')
    print(
        'Frequencies: ',
        mech_freq,
        '\nAmplitudes: ',
        mech_amp,
        '\nforward speed: ',
        speed_forward,
        '\nlateral speed: ',
        speed_lateral,
        '\nEnergy: ',
        energy,
        '\nCoT: ',
        cot)

    # pylog.warning("TODO: 1.2: Plot joint angles + CoM trajectory")

    joints_names_decoded = [name.decode('utf-8') for name in joints_names]
    
    indices_actifs  = list(range(8)) # Dos (0 à 7)
    indices_passifs = [16, 17]       # Pitch passifs (16 et 17)
    
    noms_actifs  = [joints_names_decoded[i] for i in indices_actifs]
    noms_passifs = [joints_names_decoded[i] for i in indices_passifs]

    times = sim_times[:sensor_data_joints_positions.shape[0]]
    mask_6s = times <= 6.0

    # L'astuce pour avoir les vraies 10 couleurs de base sans en sauter aucune !
    colors = plt.cm.tab10.colors 

    # ---- 2. Dessin des graphiques ----
    fig1, axs = plt.subplots(3, 1, figsize=(14, 12))
    fig1.suptitle('Joint angles over time (0–6 s)')

    # Active joints 0-3
    for i in range(4):
        idx = indices_actifs[i]
        axs[0].plot(times[mask_6s], sensor_data_joints_positions[mask_6s, idx],
                    color=colors[i], label=f'{noms_actifs[i]}')
    axs[0].set_title('Active Joints (0 to 3)')
    axs[0].set_ylabel('Angle [rad]')
    axs[0].set_xlim(0.0, 6.0)
    axs[0].legend(fontsize=8, ncol=2)
    axs[0].grid(True)

    # Active joints 4-7
    for i in range(4, 8):
        idx = indices_actifs[i]
        axs[1].plot(times[mask_6s], sensor_data_joints_positions[mask_6s, idx],
                    color=colors[i], label=f'{noms_actifs[i]}')
    axs[1].set_title('Active Joints (4 to 7)')
    axs[1].set_ylabel('Angle [rad]')
    axs[1].set_xlim(0.0, 6.0)
    axs[1].legend(fontsize=8, ncol=2)
    axs[1].grid(True)

    # Passive joints (Vérifie qu'elles existent bien dans le tableau)
    if sensor_data_joints_positions.shape[1] > 17:
        for i in range(len(indices_passifs)):
            idx = indices_passifs[i]
            axs[2].plot(times[mask_6s], sensor_data_joints_positions[mask_6s, idx],
                        color=colors[i], label=f'{noms_passifs[i]}')
        axs[2].legend(fontsize=8, ncol=2)
    else:
        axs[2].text(0.5, 0.5, "Données passives non disponibles", ha='center', va='center')
        
    axs[2].set_title('Passive Pitch Joints (Indices 16 & 17)')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Angle [rad]')
    axs[2].set_xlim(0.0, 6.0)
    axs[2].grid(True)

    fig1.tight_layout()
    fig1.savefig(os.path.join(PLOT_PATH, "joint_angles_0_6s_1_2.png"), dpi=150)
    plt.show()
    # Compute CoM (mean over links)
    com_positions = sensor_data_links_positions.mean(axis=1)  # shape: (time, 3)

    # Extract x and y (horizontal plane)
    com_x = com_positions[:, 0]
    com_y = com_positions[:, 1]

    # Plot trajectory
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(com_x, com_y)
    ax2.set_xlabel("X position [m]")
    ax2.set_ylabel("Y position [m]")
    ax2.set_title("Center of Mass Trajectory")
    ax2.axis('equal')
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_PATH, "com_trajectory_1_2.png"), dpi=150)
    plt.show()


def main(**kwargs):
    """ex1.1 main"""
    os.makedirs(PLOT_PATH, exist_ok=True)
    controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {'freq': 2.0,
                   'twl': 1.0,
                   'amp': 2.0}
    }

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording=None,
    )

    post_processing()


def exercise1_1(**kwargs):
    """Entry point for exercise 1.1 with optional plotting and runtime flags."""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise1_1(plot=True)