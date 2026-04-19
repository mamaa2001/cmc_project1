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

######################## début code Estelle #################################################
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
        joints_array_with = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
        joints_names = f['FARMSLISTanimats']['0']['sensors']['joints']['names'][:]
        #print("timestep 100, all joints, feature 0:", joints_array_with[100, :, 0])  # positions?
        #print("timestep 100, all joints, feature 1:", joints_array_with[100, :, 1])  # velocities?
        #print("timestep 100, all joints, feature 11:", joints_array_with[100, :, 11]) # torques?
        #print("timestep 100, all joints, feature 14:", joints_array_with[100, :, 14])
    with h5py.File(os.path.join(BASE_PATH, 'simulation_without_sf.hdf5'), 'r') as f:
        sensor_data_links_without = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        joints_array_without = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    #print("links array shape:", sensor_data_links_with.shape)

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

    ######## Forward speed ######
    #links_positions_with  = sensor_data_links_with[:, :, :3]
    links_positions_with  = sensor_data_links_with[:, :, 7:10]
    links_velocities_with = sensor_data_links_with[:, :, 14:17]

    #links_positions_without  = sensor_data_links_without[:, :, :3]
    links_positions_without  = sensor_data_links_without[:, :, 7:10]
    links_velocities_without = sensor_data_links_without[:, :, 14:17]

    # CoM trajectory
    #com_positions_with    = sensor_data_links_with.mean(axis=1)
    #com_positions_without = sensor_data_links_without.mean(axis=1)
    com_positions_with    = links_positions_with.mean(axis=1)
    com_positions_without = links_positions_without.mean(axis=1)
    com_x_with    = com_positions_with[:, 0]
    com_y_with    = com_positions_with[:, 1]
    com_x_without = com_positions_without[:, 0]
    com_y_without = com_positions_without[:, 1]

    # Joints
    joint_angles_with    = joints_array_with[:, :8, 0]    # (n_iterations, 8)
    joint_angles_without = joints_array_without[:, :8, 0]
    
    joints_velocities_with    = joints_array_with[:, :, 1]
    joints_torques_with       = joints_array_with[:, :, 11]
    joints_velocities_without = joints_array_without[:, :, 1]
    joints_torques_without    = joints_array_without[:, :, 11]

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
    #plt.show()

    # ---- Figure 4: CoM trajectory ----

    joints_to_plot = list(range(8))  # subset for clarity
    colors = plt.cm.viridis(np.linspace(0, 1, len(joints_to_plot)))

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle('Body joint angles: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    for idx, j in enumerate(joints_to_plot):
        axes4[0].plot(t, joint_angles_with[mask, j], color=colors[idx], label=f'joint {j}')
    axes4[0].set_title('Joint angles — with stretch feedback')
    axes4[0].set_xlabel('Time (s)')
    axes4[0].set_ylabel('Angle (rad)')
    axes4[0].legend(fontsize=7)
    axes4[0].grid()

    for idx, j in enumerate(joints_to_plot):
        axes4[1].plot(t, joint_angles_without[mask, j], color=colors[idx], label=f'joint {j}')
    axes4[1].set_title('Joint angles — without stretch feedback')
    axes4[1].set_xlabel('Time (s)')
    axes4[1].set_ylabel('Angle (rad)')
    axes4[1].legend(fontsize=7)
    axes4[1].grid()

    fig4.tight_layout()
    fig4.savefig(os.path.join(PLOT_PATH, 'joint_angles.png'), dpi=150)
    plt.show()

    ########### Neural metrics #########

    # skip transient (first 2s)
    transient = int(2.0 / timestep)
    t_steady = time[transient:]

    # compute metrics
    peak_freq_with, _, peak_amp_with = compute_frequency_amplitude_fft(
        t_steady, motor_diff_with[transient:, :])
    peak_freq_without, _, peak_amp_without = compute_frequency_amplitude_fft(
        t_steady, motor_diff_without[transient:, :])

    neural_freq_with    = np.mean(peak_freq_with)
    neural_amp_with    = np.mean(peak_amp_with)
    neural_freq_without = np.mean(peak_freq_without)
    neural_amp_without = np.mean(peak_amp_without)

    print("=== Neural Metrics ===")
    print(f"With stretch feedback:    neural_freq={neural_freq_with:.3f} Hz,  neural_amp={ neural_amp_with:.3f}")
    print(f"Without stretch feedback: neural_freq={ neural_freq_without:.3f} Hz, neural_amp={ neural_amp_without:.3f}")

    ######## Forward speed ############
    
    forward_speed_with,  _ = compute_mechanical_speed(links_positions_with,  links_velocities_with)
    forward_speed_without, _ = compute_mechanical_speed(links_positions_without, links_velocities_without)
    print("=== Other Metrics ===")
    print("Forward speed")
    print(f"With stretch feedback:    forward_speed={forward_speed_with:.3f}")
    print(f"Without stretch feedback: forward_speed={forward_speed_without:.3f}")

    ######### CoT ############
    
    _, CoT_with = compute_mechanical_energy_and_cot(t_steady, links_positions_with[transient:],
        joints_torques_with[transient:],
        joints_velocities_with[transient:]
    )
    _, CoT_without = compute_mechanical_energy_and_cot(t_steady, links_positions_without[transient:],
        joints_torques_without[transient:],
        joints_velocities_without[transient:]
    )
    print("CoT")
    print(f"With stretch feedback:    CoT={CoT_with:.3f}") 
    print(f"Without stretch feedback: CoT={CoT_without:.3f}")
    

    
######################## fin code Estelle #################################################


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
    headless = kwargs.pop('headless', True) #Si True ne fait pas la video

    #pylog.warning("TODO: 3.1 Simulate with and without sensory feedback")

    #pylog.warning("TODO: 3.1 Compare the performance")

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=w_ipsi,
        recording=None,#'animation3_1_with_sf.mp4',
        hdf5_name='simulation_with_sf.hdf5',
        controller_name='controller_with_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=0,
        recording=None, #'animation3_1_without_sf.mp4',
        hdf5_name='simulation_without_sf.hdf5',
        controller_name='controller_without_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )
  
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

