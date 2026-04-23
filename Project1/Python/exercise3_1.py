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

######################################### début code Estelle #################################################

def post_processing_3_1():

    with open(os.path.join(BASE_PATH, 'controller_with_sf.pkl'), 'rb') as f:
        controller_with = pickle.load(f)
    with open(os.path.join(BASE_PATH, 'controller_without_sf.pkl'), 'rb') as f:
        controller_without = pickle.load(f)

    with h5py.File(os.path.join(BASE_PATH, 'simulation_with_sf.hdf5'), 'r') as f:
        sensor_data_links_with = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        joints_array_with = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
        joints_names = f['FARMSLISTanimats']['0']['sensors']['joints']['names'][:]  

    with h5py.File(os.path.join(BASE_PATH, 'simulation_without_sf.hdf5'), 'r') as f:
        sensor_data_links_without = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        joints_array_without = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]


    state_with = controller_with['state']
    state_without = controller_without['state']

    timestep = 0.004
    n_iterations = state_with.shape[0]
    time = np.arange(n_iterations) * timestep

    # Only plot first 5 seconds
    t_end = 5.0
    mask = time <= t_end
    t = time[mask]

    
    joints = list(range(8))   # which joints to plot
    #colors = plt.cm.viridis(np.linspace(0, 1, len(joints)))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

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
    axes1[0, 0].grid()

    # theta without
    for idx, j in enumerate(joints):
        axes1[0, 1].plot(t, theta_left_without[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[0, 1].plot(t, theta_right_without[mask, j], color=colors[idx], linestyle='--',label=f'joint {j} right')
    axes1[0, 1].set_title('Phase θ — without stretch feedback')
    axes1[0, 1].set_xlabel('Time (s)')
    axes1[0, 1].set_ylabel('Phase (rad)')
    axes1[0, 1].legend(fontsize=7)
    axes1[0, 1].grid()

    # r with
    for idx, j in enumerate(joints):
        axes1[1, 0].plot(t, r_left_with[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[1, 0].plot(t, r_right_with[mask, j], color=colors[idx], linestyle='--',label=f'joint {j} right')
    axes1[1, 0].set_title('Amplitude r — with stretch feedback')
    axes1[1, 0].set_xlabel('Time (s)')
    axes1[1, 0].set_ylabel('Amplitude')
    axes1[1, 0].legend(fontsize=7)
    axes1[1, 0].grid()

    # r without
    for idx, j in enumerate(joints):
        axes1[1, 1].plot(t, r_left_without[mask, j],  color=colors[idx], linestyle='-', label=f'joint {j} left')
        axes1[1, 1].plot(t, r_right_without[mask, j], color=colors[idx], linestyle='--',label=f'joint {j} right')
    axes1[1, 1].set_title('Amplitude r — without stretch feedback')
    axes1[1, 1].set_xlabel('Time (s)')
    axes1[1, 1].set_ylabel('Amplitude')
    axes1[1, 1].legend(fontsize=7)
    axes1[1, 1].grid()

    fig1.tight_layout()
    os.makedirs(PLOT_PATH, exist_ok=True)
    fig1.savefig(os.path.join(PLOT_PATH, 'oscillator_states_theta_r.png'), dpi=150)

    #############################################################################################################

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


    # ---- Figure 2: motor sum and diff ----
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Motor outputs: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    # sum with
    for idx, j in enumerate(joints):
        axes2[0, 0].plot(t, motor_sum_with[mask, j], color=colors[idx], label=f'jsum_{j}')
    axes2[0, 0].set_title('Motor sum (ML+MR) — with stretch feedback')
    axes2[0, 0].set_xlabel('Time (s)')
    axes2[0, 0].set_ylabel('ML + MR')
    axes2[0, 0].legend(fontsize=7)
    axes2[0, 0].grid()

    # sum without
    for idx, j in enumerate(joints):
        axes2[0, 1].plot(t, motor_sum_without[mask, j], color=colors[idx], label=f'sum_{j}')
    axes2[0, 1].set_title('Motor sum (ML+MR) — without stretch feedback')
    axes2[0, 1].set_xlabel('Time (s)')
    axes2[0, 1].set_ylabel('ML + MR')
    axes2[0, 1].legend(fontsize=7)
    axes2[0, 1].grid()

    # diff with
    for idx, j in enumerate(joints):
        axes2[1, 0].plot(t, motor_diff_with[mask, j], color=colors[idx], label=f'diff_{j}')
    axes2[1, 0].set_title('Motor diff (ML-MR) — with stretch feedback')
    axes2[1, 0].set_xlabel('Time (s)')
    axes2[1, 0].set_ylabel('ML - MR')
    axes2[1, 0].legend(fontsize=7)
    axes2[1, 0].grid()

    # diff without
    for idx, j in enumerate(joints):
        axes2[1, 1].plot(t, motor_diff_without[mask, j], color=colors[idx], label=f'diff_{j}')
    axes2[1, 1].set_title('Motor diff (ML-MR) — without stretch feedback')
    axes2[1, 1].set_xlabel('Time (s)')
    axes2[1, 1].set_ylabel('ML - MR')
    axes2[1, 1].legend(fontsize=7)
    axes2[1, 1].grid()

    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOT_PATH, 'oscillator_states_motor.png'), dpi=150)

    #############################################################################################################
    
    ######## Forward speed ######
    links_positions_with  = sensor_data_links_with[:, :, 7:10]
    links_velocities_with = sensor_data_links_with[:, :, 14:17]

   
    links_positions_without  = sensor_data_links_without[:, :, 7:10]
    links_velocities_without = sensor_data_links_without[:, :, 14:17]

  
    # Joints
    #joint_angles_with    = joints_array_with[:, :8, 0]    # (n_iterations, 8)
    #joint_angles_without = joints_array_without[:, :8, 0]
    
    joints_names_decoded = [name.decode('utf-8') for name in joints_names]
    indices_actifs  = list(range(8))
    indices_passifs = [16, 17]
    noms_actifs  = [joints_names_decoded[i] for i in indices_actifs]
    noms_passifs = [joints_names_decoded[i] for i in indices_passifs]

    joint_angles_with_active    = joints_array_with[:, indices_actifs, 0]
    joint_angles_with_passive   = joints_array_with[:, indices_passifs, 0]
    joint_angles_without_active  = joints_array_without[:, indices_actifs, 0]
    joint_angles_without_passive = joints_array_without[:, indices_passifs, 0]

    
    joints_velocities_with    = joints_array_with[:, :, 1]
    joints_torques_with       = joints_array_with[:, :, 11]
    joints_velocities_without = joints_array_without[:, :, 1]
    joints_torques_without    = joints_array_without[:, :, 11]

 
    #############################################################################################################
    
    # CoM trajectory
    com_positions_with    = links_positions_with.mean(axis=1)
    com_positions_without = links_positions_without.mean(axis=1)
    com_x_with    = com_positions_with[:, 0]
    com_y_with    = com_positions_with[:, 1]
    com_x_without = com_positions_without[:, 0]
    com_y_without = com_positions_without[:, 1]

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

    #############################################################################################################
    
    '''
    # ---- Figure 4: ----

    joints_to_plot = list(range(8))  # subset for clarity
    #colors = plt.cm.viridis(np.linspace(0, 1, len(joints_to_plot)))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle('Body joint angles: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    for idx, j in enumerate(joints_to_plot):
        axes4[0].plot(t, joint_angles_with[mask, j], color=colors[idx], label=f'Joint {j}')
    axes4[0].set_title('Joint angles — with stretch feedback')
    axes4[0].set_xlabel('Time [s]')
    axes4[0].set_ylabel('Angle [rad]')
    axes4[0].legend(fontsize=7)
    axes4[0].grid()

    for idx, j in enumerate(joints_to_plot):
        axes4[1].plot(t, joint_angles_without[mask, j], color=colors[idx], label=f'Joint {j}')
    axes4[1].set_title('Joint angles — without stretch feedback')
    axes4[1].set_xlabel('Time [s]')
    axes4[1].set_ylabel('Angle [rad]')
    axes4[1].legend(fontsize=7)
    axes4[1].grid()

    fig4.tight_layout()
    fig4.savefig(os.path.join(PLOT_PATH, 'joint_angles.png'), dpi=150)
    plt.show()
    '''
    # Avec passive joints
    
    # V1
    # ---- Figure 4: Joint angles ----
    '''
    joints_names_decoded = [name.decode('utf-8') for name in joints_names]
    indices_actifs  = list(range(8))
    indices_passifs = [16, 17]
    noms_actifs  = [joints_names_decoded[i] for i in indices_actifs]
    noms_passifs = [joints_names_decoded[i] for i in indices_passifs]

    joint_angles_with_active    = joints_array_with[:, indices_actifs, 0]
    joint_angles_with_passive   = joints_array_with[:, indices_passifs, 0]
    joint_angles_without_active  = joints_array_without[:, indices_actifs, 0]
    joint_angles_without_passive = joints_array_without[:, indices_passifs, 0]
    '''
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    fig4, axs = plt.subplots(3, 1, figsize=(14, 12))
    fig4.suptitle('Joint angles: with (—) vs without (--) stretch feedback')

    # Active joints 0-3
    for i in range(4):
        axs[0].plot(t, joint_angles_with_active[mask, i],    color=colors[i], linestyle='-',  label=f'{noms_actifs[i]} with')
        axs[0].plot(t, joint_angles_without_active[mask, i], color=colors[i], linestyle='--', label=f'{noms_actifs[i]} without')
    axs[0].set_title('Active Joints 0-3')
    axs[0].set_ylabel('Angle [rad]')
    axs[0].legend(fontsize=7, ncol=2)
    axs[0].grid(True)

    # Active joints 4-7
    for i in range(4, 8):
        axs[1].plot(t, joint_angles_with_active[mask, i],    color=colors[i], linestyle='-',  label=f'{noms_actifs[i]} with')
        axs[1].plot(t, joint_angles_without_active[mask, i], color=colors[i], linestyle='--', label=f'{noms_actifs[i]} without')
    axs[1].set_title('Active Joints 4-7')
    axs[1].set_ylabel('Angle [rad]')
    axs[1].legend(fontsize=7, ncol=2)
    axs[1].grid(True)

    # Passive joints
    for i in range(2):
        axs[2].plot(t, joint_angles_with_passive[mask, i],    color=colors[i], linestyle='-',  label=f'{noms_passifs[i]} with')
        axs[2].plot(t, joint_angles_without_passive[mask, i], color=colors[i], linestyle='--', label=f'{noms_passifs[i]} without')
    axs[2].set_title('Passive Joints')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Angle [rad]')
    axs[2].legend(fontsize=7, ncol=2)
    axs[2].grid(True)

    fig4.tight_layout()
    fig4.savefig(os.path.join(PLOT_PATH, 'joint_angles.png'), dpi=150)
    '''

    # V2
    # ---- Figure 4: Joint angles (active + passive) ----

    fig4, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig4.suptitle('Joint angles: with (w_ipsi=3) vs without (w_ipsi=0) stretch feedback')

    for col, (active, passive, label) in enumerate([
        (joint_angles_with_active,    joint_angles_with_passive,    'with stretch feedback'),
        (joint_angles_without_active, joint_angles_without_passive, 'without stretch feedback'),
    ]):
        all_angles = np.concatenate([active, passive], axis=1)
        noms_all   = noms_actifs + noms_passifs

        # Active joints
        for i, name in enumerate(noms_actifs):
            axs[0, col].plot(t, active[mask, i], label=name)
        axs[0, col].set_title(f'Active Joints — {label}')
        axs[0, col].set_ylabel('Angle [rad]')
        axs[0, col].legend(fontsize=7)
        axs[0, col].grid(True)

        # Passive joints
        for i, name in enumerate(noms_passifs):
            axs[1, col].plot(t, passive[mask, i], label=name)
        axs[1, col].set_title(f'Passive Joints — {label}')
        axs[1, col].set_ylabel('Angle [rad]')
        axs[1, col].legend(fontsize=7)
        axs[1, col].grid(True)

        # All joints
        for i, name in enumerate(noms_all):
            axs[2, col].plot(t, all_angles[mask, i], label=name)
        axs[2, col].set_title(f'All Joints — {label}')
        axs[2, col].set_xlabel('Time [s]')
        axs[2, col].set_ylabel('Angle [rad]')
        axs[2, col].legend(fontsize=7)
        axs[2, col].grid(True)

    fig4.tight_layout()
    fig4.savefig(os.path.join(PLOT_PATH, 'joint_angles.png'), dpi=150)
    '''
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

    ########## CoT ############
    
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
        runtime_n_iterations=1001,
        runtime_buffer_size=1001,
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
        runtime_n_iterations=1001,
        runtime_buffer_size=1001,
        fast=fast,
        headless=headless,
    )
  
    post_processing_3_1()




  
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

