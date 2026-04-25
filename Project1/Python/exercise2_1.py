#!/usr/bin/env python3

import time
import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt


from farms_core import pylog
from farms_core.utils.profile import profile

from simulate import runsim
from cmc_controllers.metrics import filter_signals

BASE_PATH = 'logs/exercise2_1/'
PLOT_PATH = 'results'


def post_processing(base_path):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
        joints_names = f['FARMSLISTanimats']['0']['sensors']['joints']['names'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    # Load Controller
    with open(base_path + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)
    print("controleur data keys:", controller_data.keys())

    

    print(joints_names) 

    state = controller_data['state']  # (2501, 48)

    # Colonnes 0-15 : phases θ entrelacées (gauche=pairs, droite=impairs)
    theta_left  = state[:, slice(0, 16, 2)]   # cols 0,2,4,6,8,10,12,14  → 8 oscillateurs gauche
    theta_right = state[:, slice(1, 17, 2)]   # cols 1,3,5,7,9,11,13,15  → 8 oscillateurs droite

    # Colonnes 16-31 : amplitudes r 
    r_left  = state[:, slice(16, 32, 2)]      # cols 16,18,20,22,24,26,28,30
    r_right = state[:, slice(17, 33, 2)]      # cols 17,19,21,23,25,27,29,31

    motor_left  = state[:, slice(32, 48, 2)]  # cols 32,34,36,38,40,42,44,46
    motor_right = state[:, slice(33, 49, 2)]  # cols 33,35,37,39,41,43,45,47    
    # Sum et diff
    motor_sum  = motor_left + motor_right   # mouvement symétrique
    motor_diff = motor_left - motor_right   # mouvement antisymétrique

    mask   = sim_times <= 5.0
    t_plot = sim_times[mask]

    # Plot θ
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    plt.figure()
    for i in range(8):
        c = colors[i]
        plt.plot(t_plot, theta_left[mask, i],  label=f"θ_L{i}", color=c)
        plt.plot(t_plot, theta_right[mask, i], label=f"θ_R{i}", linestyle='--', color=c)

    plt.xlabel("Time [s]")
    plt.ylabel("Phase θ")
    plt.title("Time evolution of phases (θ)")
    plt.legend(ncol=2)
    plt.show()

    # Plot r
    plt.figure()
    for i in range(8):
        plt.plot(t_plot, r_left[mask, i],  label=f"r_L{i}")
        plt.plot(t_plot, r_right[mask, i], label=f"r_R{i}", linestyle='--')
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude r")
    plt.title("Time evolution of amplitudes (r)"); plt.legend(); plt.grid()

    # Plot sum
    plt.figure()
    for i in range(8):
        plt.plot(t_plot, motor_sum[mask, i], label=f"sum_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Sum L+R")
    plt.title("Motor output sum (L+R)"); plt.legend(); plt.grid()

    # Plot diff
    plt.figure()
    for i in range(8):
        plt.plot(t_plot, motor_diff[mask, i], label=f"diff_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Diff L-R")
    plt.title("Motor output difference (L-R)"); plt.legend(); plt.grid()

    #########################################################################

    ################ Body Joints ##################
    min_len = min(len(sim_times), sensor_data_joints_positions.shape[0])

    times = sim_times[:min_len]

    joints_names_decoded = [name.decode('utf-8') for name in joints_names]

    indices_actifs = list(range(8))
    indices_passifs = [16, 17]

    joints_actifs = sensor_data_joints_positions[:min_len, indices_actifs]
    joints_passifs = sensor_data_joints_positions[:min_len, indices_passifs]

    ### In degrees ###
    joints_actifs = joints_actifs * 180 / np.pi
    joints_passifs = joints_passifs * 180 / np.pi


    noms_actifs = [joints_names_decoded[i] for i in indices_actifs]
    noms_passifs = [joints_names_decoded[i] for i in indices_passifs]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Active Joints
    for i, idx in enumerate(indices_actifs):
        axs[0].plot(times[:min_len], joints_actifs[:, i], label=noms_actifs[i])
    axs[0].set_title('Active Joints')
    axs[0].set_ylabel('Angle [deg]')
    axs[0].legend()
    axs[0].grid(True)

    # Passive Joints
    for i, idx in enumerate(indices_passifs):
        axs[1].plot(times[:min_len], joints_passifs[:, i], label=noms_passifs[i])
    axs[1].set_title('Passive Joints')
    axs[1].set_ylabel('Angle [deg]')
    axs[1].legend()
    axs[1].grid(True)

    # All Joints
    joints_all = np.concatenate([joints_actifs, joints_passifs], axis=1)
    noms_all = noms_actifs + noms_passifs
    for i in range(joints_all.shape[1]):
        axs[2].plot(times[:min_len], joints_all[:, i], label=noms_all[i])
    axs[2].set_title('All Joints')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Angle [deg]')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    ###################### Center of Mass Trajectory ##################

    com_positions = sensor_data_links_positions.mean(axis=1)  # shape: (time, 3)

    # Extract x and y (horizontal plane)
    com_x = com_positions[:, 0]
    com_y = com_positions[:, 1]

    # Plot trajectory
    plt.figure()
    plt.plot(com_x, com_y)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Center of Mass Trajectory")
    plt.axis('equal')
    plt.grid()
    plt.show()
    
    #########################################################################


def main(**kwargs):
    """Run exercise 2.1 simulation and post-processing pipeline."""
    os.makedirs(PLOT_PATH, exist_ok=True)
    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3,
            'drive_right': 3,
            'd_low': 1,
            'd_high': 5,
            'a_rate': np.ones(8) * 3,
            'offset_freq': np.ones(8) * 1,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': np.ones(7) * np.pi * 2 / 8,
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(
                seed=42).uniform(
                0.0,
                2 * np.pi,
                size=16)}}

    tic = time.time()
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording='exercise2_1.mp4',
    )
    post_processing(BASE_PATH)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


def exercise2_1(**kwargs):
    """ex2.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()

if __name__ == '__main__':
    exercise2_1(plot=True)

