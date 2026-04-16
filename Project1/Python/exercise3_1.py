#!/usr/bin/env python3

import time
import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.utils.profile import profile

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
)
from simulate import runsim


BASE_PATH = 'logs/exercise3_1/'
PLOT_PATH = 'results'

'''
def post_processing_with_sf(base_path):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + 'simulation_with_sf.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        state_data = f['FARMSLISTanimats']['0']['state'][:] #rajout pour les plots
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    # Load Controller
    with open(base_path + "controller_with_sf.pkl", "rb") as f:
        controller_data = pickle.load(f)
    print("controleur data keys:", controller_data.keys())

    # pour l'instant c'est guez de ouf

    # θ et r directement depuis HDF5
    theta_hist = sensor_data_links_positions   
    r_hist     = sensor_data_joints_positions  

    # Filtre 5s
    mask = sim_times <= 5.0
    t = sim_times[mask]

    # Plot θ
    plt.figure()
    for i in range(16):
        plt.plot(t, theta_hist[mask, i], label=f"θ_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Phase θ")
    plt.title("Time evolution of phases (θ) WITH SF"); plt.legend(); plt.grid()

    # Plot r
    plt.figure()
    for i in range(16):
        plt.plot(t, r_hist[mask, i], label=f"r_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude r")
    plt.title("Time evolution of amplitudes (r) WITH SF"); plt.legend(); plt.grid()

    #########################################################################

def post_processing_without_sf(base_path):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + 'simulation_without_sf.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        state_data = f['FARMSLISTanimats']['0']['state'][:] #rajout pour les plots
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    # Load Controller
    with open(base_path + "controller_without_sf.pkl", "rb") as f:
        controller_data = pickle.load(f)
    print("controleur data keys:", controller_data.keys())

    # pour l'instant c'est guez de ouf

    # θ et r directement depuis HDF5
    theta_hist = state_data[:, :16]   # colonnes 0-15
    r_hist     = state_data[:, 16:32] # colonnes 16-31

    # Filtre 5s
    mask = sim_times <= 5.0
    t = sim_times[mask]

    # Plot θ
    plt.figure()
    for i in range(16):
        plt.plot(t, theta_hist[mask, i], label=f"θ_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Phase θ")
    plt.title("Time evolution of phases (θ) NO SF"); plt.legend(); plt.grid()

    # Plot r
    plt.figure()
    for i in range(16):
        plt.plot(t, r_hist[mask, i], label=f"r_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude r")
    plt.title("Time evolution of amplitudes (r) NO SF"); plt.legend(); plt.grid()

    #########################################################################

'''
'''
def plot_oscillator_states():
    """Plot oscillator states (theta and r) for w_ipsi=3 and w_ipsi=0"""

    with open(os.path.join(BASE_PATH, 'controller_with_sf.pkl'), 'rb') as f:
        controller_with = pickle.load(f)
    with open(os.path.join(BASE_PATH, 'controller_without_sf.pkl'), 'rb') as f:
        controller_without = pickle.load(f)

    # Extract states directly from the dict
    state_with = controller_with['state']       # shape (n_iterations, 3*n_oscillators)
    state_without = controller_without['state']

    n_oscillators = 16
    n_iterations = state_with.shape[0]
    timestep = 0.004  # adjust if different
    time = np.arange(n_iterations) * timestep

    # Extract theta and r
    theta_with    = state_with[:, :n_oscillators]
    r_with        = state_with[:, n_oscillators:2*n_oscillators]
    theta_without = state_without[:, :n_oscillators]
    r_without     = state_without[:, n_oscillators:2*n_oscillators]
    motor
    # Only plot first 5 seconds
    t_end = 5.0
    mask = time <= t_end

    # Plot a subset of oscillators for clarity (e.g. left chain: 0,2,4,6)
    osc_to_plot = [0, 2, 4, 6]
    colors = plt.cm.viridis(np.linspace(0, 1, len(osc_to_plot)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Oscillator states: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    # --- Theta with feedback ---
    ax = axes[0, 0]
    for idx, osc in enumerate(osc_to_plot):
        ax.plot(time[mask], theta_with[mask, osc], color=colors[idx], label=f'osc {osc}')
    ax.set_title('Phase θ — with stretch feedback')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()

    # --- Theta without feedback ---
    ax = axes[0, 1]
    for idx, osc in enumerate(osc_to_plot):
        ax.plot(time[mask], theta_without[mask, osc], color=colors[idx], label=f'osc {osc}')
    ax.set_title('Phase θ — without stretch feedback')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase (rad)')
    ax.legend()

    # --- r with feedback ---
    ax = axes[1, 0]
    for idx, osc in enumerate(osc_to_plot):
        ax.plot(time[mask], r_with[mask, osc], color=colors[idx], label=f'osc {osc}')
    ax.set_title('Amplitude r — with stretch feedback')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()

    # --- r without feedback ---
    ax = axes[1, 1]
    for idx, osc in enumerate(osc_to_plot):
        ax.plot(time[mask], r_without[mask, osc], color=colors[idx], label=f'osc {osc}')
    ax.set_title('Amplitude r — without stretch feedback')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()

    plt.tight_layout()
    os.makedirs(PLOT_PATH, exist_ok=True)
    plt.savefig(os.path.join(PLOT_PATH, 'oscillator_states_comparison.png'), dpi=150)
    plt.show()
'''
def plot_oscillator_states():

    with open(os.path.join(BASE_PATH, 'controller_with_sf.pkl'), 'rb') as f:
        controller_with = pickle.load(f)
    with open(os.path.join(BASE_PATH, 'controller_without_sf.pkl'), 'rb') as f:
        controller_without = pickle.load(f)

    with h5py.File(os.path.join(BASE_PATH, 'simulation_with_sf.hdf5'), 'r') as f:
        sensor_data_links_with = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
    with h5py.File(os.path.join(BASE_PATH, 'simulation_without_sf.hdf5'), 'r') as f:
        sensor_data_links_without = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]

    print("links array shape:", sensor_data_links_with.shape)

    state_with = controller_with['state']
    state_without = controller_without['state']

    timestep = 0.004
    n_iterations = state_with.shape[0]
    time = np.arange(n_iterations) * timestep

    # Extract phases
    theta_left_with   = state_with[:, slice(0, 16, 2)]
    theta_right_with  = state_with[:, slice(1, 17, 2)]
    theta_left_without   = state_without[:, slice(0, 16, 2)]
    theta_right_without  = state_without[:, slice(1, 17, 2)]

    # Extract amplitudes
    r_left_with   = state_with[:, slice(16, 32, 2)]
    r_right_with  = state_with[:, slice(17, 33, 2)]
    r_left_without   = state_without[:, slice(16, 32, 2)]
    r_right_without  = state_without[:, slice(17, 33, 2)]

    # Extract motor outputs
    motor_left_with   = state_with[:, slice(32, 48, 2)]
    motor_right_with  = state_with[:, slice(33, 49, 2)]
    motor_left_without   = state_without[:, slice(32, 48, 2)]
    motor_right_without  = state_without[:, slice(33, 49, 2)]

    # Sum and diff
    motor_sum_with   = motor_left_with  + motor_right_with
    motor_diff_with  = motor_left_with  - motor_right_with
    motor_sum_without  = motor_left_without + motor_right_without
    motor_diff_without = motor_left_without - motor_right_without

    # CoM trajectory
    com_positions_with    = sensor_data_links_with.mean(axis=1)
    com_positions_without = sensor_data_links_without.mean(axis=1)
    com_x_with    = com_positions_with[:, 0]
    com_y_with    = com_positions_with[:, 1]
    com_x_without = com_positions_without[:, 0]
    com_y_without = com_positions_without[:, 1]

    # Only plot first 5 seconds
    t_end = 5.0
    mask = time <= t_end
    t = time[mask]

    joints = [0, 1, 2, 3]  # which joints to plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(joints)))

  
    # ---- Figure 1: theta and r ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Oscillator states: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    # theta with
    for idx, j in enumerate(joints):
        axes1[0, 0].plot(t, theta_left_with[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[0, 0].plot(t, theta_right_with[mask, j], color=colors[idx], linestyle='--', label=f'joint {j} right')
    axes1[0, 0].set_title('Phase θ — with stretch feedback')
    axes1[0, 0].set_xlabel('Time (s)')
    axes1[0, 0].set_ylabel('Phase (rad)')
    axes1[0, 0].legend(fontsize=7)

    # theta without
    for idx, j in enumerate(joints):
        axes1[0, 1].plot(t, theta_left_without[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[0, 1].plot(t, theta_right_without[mask, j], color=colors[idx], linestyle='--',label=f'joint {j} right')
    axes1[0, 1].set_title('Phase θ — without stretch feedback')
    axes1[0, 1].set_xlabel('Time (s)')
    axes1[0, 1].set_ylabel('Phase (rad)')
    axes1[0, 1].legend(fontsize=7)

    # r with
    for idx, j in enumerate(joints):
        axes1[1, 0].plot(t, r_left_with[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[1, 0].plot(t, r_right_with[mask, j], color=colors[idx], linestyle='--',label=f'joint {j} right')
    axes1[1, 0].set_title('Amplitude r — with stretch feedback')
    axes1[1, 0].set_xlabel('Time (s)')
    axes1[1, 0].set_ylabel('Amplitude')
    axes1[1, 0].legend(fontsize=7)

    # r without
    for idx, j in enumerate(joints):
        axes1[1, 1].plot(t, r_left_without[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[1, 1].plot(t, r_right_without[mask, j], color=colors[idx], linestyle='--',label=f'joint {j} right')
    axes1[1, 1].set_title('Amplitude r — without stretch feedback')
    axes1[1, 1].set_xlabel('Time (s)')
    axes1[1, 1].set_ylabel('Amplitude')
    axes1[1, 1].legend(fontsize=7)

    fig1.tight_layout()
    os.makedirs(PLOT_PATH, exist_ok=True)
    fig1.savefig(os.path.join(PLOT_PATH, 'oscillator_states_theta_r.png'), dpi=150)

    # ---- Figure 2: motor sum and diff ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Motor outputs: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    # sum with
    for idx, j in enumerate(joints):
        axes2[0, 0].plot(t, motor_sum_with[mask, j], color=colors[idx], label=f'joint {j}')
    axes2[0, 0].set_title('Motor sum (ML+MR) — with stretch feedback')
    axes2[0, 0].set_xlabel('Time (s)')
    axes2[0, 0].set_ylabel('ML + MR')
    axes2[0, 0].legend(fontsize=7)

    # sum without
    for idx, j in enumerate(joints):
        axes2[0, 1].plot(t, motor_sum_without[mask, j], color=colors[idx], label=f'joint {j}')
    axes2[0, 1].set_title('Motor sum (ML+MR) — without stretch feedback')
    axes2[0, 1].set_xlabel('Time (s)')
    axes2[0, 1].set_ylabel('ML + MR')
    axes2[0, 1].legend(fontsize=7)

    # diff with
    for idx, j in enumerate(joints):
        axes2[1, 0].plot(t, motor_diff_with[mask, j], color=colors[idx], label=f'joint {j}')
    axes2[1, 0].set_title('Motor diff (ML-MR) — with stretch feedback')
    axes2[1, 0].set_xlabel('Time (s)')
    axes2[1, 0].set_ylabel('ML - MR')
    axes2[1, 0].legend(fontsize=7)

    # diff without
    for idx, j in enumerate(joints):
        axes2[1, 1].plot(t, motor_diff_without[mask, j], color=colors[idx], label=f'joint {j}')
    axes2[1, 1].set_title('Motor diff (ML-MR) — without stretch feedback')
    axes2[1, 1].set_xlabel('Time (s)')
    axes2[1, 1].set_ylabel('ML - MR')
    axes2[1, 1].legend(fontsize=7)

    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_PATH, 'oscillator_states_motor.png'), dpi=150)

    # ---- Figure 3: CoM trajectory ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('CoM trajectory: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    axes3[0].plot(com_x_with, com_y_with)
    axes3[0].set_title('CoM trajectory — with stretch feedback')
    axes3[0].set_xlabel('X position (m)')
    axes3[0].set_ylabel('Y position (m)')
    axes3[0].axis('equal')
    axes3[0].grid()

    axes3[1].plot(com_x_without, com_y_without)
    axes3[1].set_title('CoM trajectory — without stretch feedback')
    axes3[1].set_xlabel('X position (m)')
    axes3[1].set_ylabel('Y position (m)')
    axes3[1].axis('equal')
    axes3[1].grid()

    fig3.tight_layout()
    fig3.savefig(os.path.join(PLOT_PATH, 'com_trajectory.png'), dpi=150)
    plt.show()

def main(**kwargs):
    """Run exercise 3.1 simulations with and without sensory feedback."""
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
                size=16),
        },
    }
    w_ipsi = 3
    fast = kwargs.pop('fast', False)
    headless = kwargs.pop('headless', False)

    #pylog.warning("TODO: 3.1 Simulate with and without sensory feedback")

    #pylog.warning("TODO: 3.1 Compare the performance")

    tic1 = time.time()
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=w_ipsi,
        recording='animation3_1_with_sf.mp4',
        hdf5_name='simulation_with_sf.hdf5',
        controller_name='controller_with_sf.pkl',
        runtime_n_iterations=501,
        runtime_buffer_size=501,
        fast=fast,
        headless=headless,
    )
    #post_processing_with_sf(BASE_PATH)
    #pylog.info('Total simulation time: %s [s]', time.time() - tic1)

    tic2 = time.time()
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=0,
        recording='animation3_1_without_sf.mp4',
        hdf5_name='simulation_without_sf.hdf5',
        controller_name='controller_without_sf.pkl',
        runtime_n_iterations=501,
        runtime_buffer_size=501,
        fast=fast,
        headless=headless,
    )
    #post_processing_without_sf(BASE_PATH)
    #pylog.info('Total simulation time: %s [s]', time.time() - tic2)
    plot_oscillator_states()




  
def exercise3_1(**kwargs):
    """ex3.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise3_1(plot=True)

