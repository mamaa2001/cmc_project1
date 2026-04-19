#!/usr/bin/env python3
"""Run exercise 2.2 parameter sweeps and generate heatmaps/trajectory plots."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_trajectory_curvature,
)

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from simulate import run_multiple
MAX_WORKERS = 16  # adjust based on your hardware capabilities

# CPG parameters
BASE_PATH = 'logs/exercise2_2/'
PLOT_PATH = 'results'


def load_metrics_from_hdf5(hdf5_path):
    """Load speed and CoT metrics from an HDF5 simulation result."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    links_positions = sensor_data_links[:, :, 7:10]
    links_velocities = sensor_data_links[:, :, 14:17]
    joints_velocities = sensor_data_joints[:, :, 1]
    joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=links_positions,
        links_velocities=links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=links_positions,
        joints_torques=joints_torques,
        joints_velocities=joints_velocities,
    )
    #added by Matt
    CoM_traj = links_positions.mean(axis=1)
    timestep = sim_times[1]-sim_times[0]
    
    curvature_mean = compute_trajectory_curvature(trajectory=CoM_traj, timestep=timestep)

    return speed_forward, speed_lateral, cot, CoM_traj, curvature_mean


def exercise2_2(**kwargs):
    pylog.warning("TODO: 2.2: Explore the effect of drive parameters and body phase bias")
    # pylog.set_level('critical')

    #En cours d'implementation et copier depuis le 1.2
    os.makedirs(PLOT_PATH, exist_ok=True)
    base_controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': None,
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
    
    drive_range = np.linspace(2.0,4.0,10)
    PL_range = np.linspace(np.pi/16,3*np.pi/8,10)
    

    parameter_grid_example = {
        'drive_right': drive_range,
        'PL': PL_range
    }
    

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_example,
        common_kwargs={'fast': True, 'headless': True},
    )
    
    drive_range_2 = np.linspace(2.0,4.0,9)
    parameter_grid_example_2 = {
        'drive_right': drive_range_2,
        'drive_left': drive_range_2
    }
    
    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_example_2,
        common_kwargs={'fast': True, 'headless': True},
    )

    plot = kwargs.pop('plot', False)
    if plot:
        metrics = []
        metrics_2 = []
        for drive_val in parameter_grid_example['drive_right']:
            for PL_val in parameter_grid_example['PL']:
                sim_result = BASE_PATH + \
                f'simulation_drive_right{drive_val:0.3f}_PL{PL_val:0.3f}.hdf5'
                v_fwd,_, cot,_,_ = load_metrics_from_hdf5(sim_result)
                int_results = {
                    'drive': drive_val,
                    'phase lag': PL_val,
                    'forward_speed': v_fwd,
                    'CoT': cot
                }
                metrics.append(int_results)
        for drive_val_right in parameter_grid_example_2['drive_right']:
            for drive_val_left in parameter_grid_example_2['drive_left']:
                sim_result = BASE_PATH + \
                f'simulation_drive_right{drive_val_right:0.3f}_drive_left{drive_val_left:0.3f}.hdf5'
                _,_, _, CoM_traj, curvature_mean = load_metrics_from_hdf5(sim_result)
                int_results = {
                    'drive right': drive_val_right,
                    'drive left': drive_val_left,
                    'trajectory CoM': CoM_traj,
                    'curvature mean': curvature_mean
                }
                metrics_2.append(int_results)

        drive_vals = np.array([m['drive'] for m in metrics])
        PL_vals = np.array([m['phase lag'] for m in metrics])
        forward_speed_vals = np.array([m['forward_speed'] for m in metrics])
        cot_vals = np.array([m['CoT'] for m in metrics])

        drive_right_vals = np.array([m['drive right'] for m in metrics_2])
        drive_left_vals = np.array([m['drive left'] for m in metrics_2])
        #traj_CoM = np.array([m['trajectory CoM'] for m in metrics_2], dtype=object)
        

        fig1 = plt.figure(figsize=(7, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.scatter(drive_vals, PL_vals, forward_speed_vals, c=forward_speed_vals, cmap='viridis')
        ax1.set_xlabel('drive')
        ax1.set_ylabel('phase lag')
        ax1.set_zlabel('forward_speed')
        ax1.set_title('Forward speed')
        plt.tight_layout()

        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(drive_vals, PL_vals, cot_vals, c=cot_vals, cmap='plasma')
        ax2.set_xlabel('drive')
        ax2.set_ylabel('phase lag')
        ax2.set_zlabel('CoT')
        ax2.set_title('CoT')
        plt.tight_layout()

        # Grid of CoM trajectories (one subplot per drive_right/drive_left pair)
        unique_drive_right = np.sort(np.unique(drive_right_vals))
        unique_drive_left = np.sort(np.unique(drive_left_vals))

        unique_drive_right = np.sort(np.unique(drive_right_vals))

        fig3, axes = plt.subplots(
            3,
            3,
            figsize=(12, 10),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for i, dr in enumerate(unique_drive_right):
            ax = axes[i]
            subset = [m for m in metrics_2 if m['drive right'] == dr]

            for m in subset:
                dl = m['drive left']
                traj = m['trajectory CoM']  # shape: [T, 3]
                ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, label=f"dl={dl:.2f}")

            ax.set_title(f"drive right = {dr:.2f}", fontsize=9)
            ax.set_xlabel("CoM x")
            ax.set_ylabel("CoM y")
            ax.legend(fontsize=7, ncol=1)

        # Hide any unused subplot if unique_drive_right has fewer than 9 values
        for j in range(len(unique_drive_right), 9):
            axes[j].axis("off")

        fig3.suptitle("CoM trajectories grouped by drive right", y=1.02)
        plt.tight_layout()
        mean_cruvatures_vals = np.array([m['curvature mean'] for m in metrics_2])

        fig4 = plt.figure(figsize=(7, 6))
        ax4 = fig4.add_subplot(111, projection='3d')
        ax4.scatter(drive_right_vals, drive_left_vals, mean_cruvatures_vals, c=mean_cruvatures_vals, cmap='plasma')
        ax4.set_xlabel('drive right')
        ax4.set_ylabel('drive left')
        ax4.set_zlabel('mean curvature')
        ax4.set_title('Mean curvature')
        plt.tight_layout()

        plt.show()


if __name__ == '__main__':
    exercise2_2(plot=True)

