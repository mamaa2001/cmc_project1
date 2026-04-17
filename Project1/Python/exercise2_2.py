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
MAX_WORKERS = 8  # adjust based on your hardware capabilities

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

    return speed_forward, speed_lateral, cot


def exercise2_2(**kwargs):
    pylog.warning("TODO: 2.2: Explore the effect of drive parameters and body phase bias")
    # pylog.set_level('critical')

    #En cours d'implementation et copier depuis le 1.2
    '''os.makedirs(PLOT_PATH, exist_ok=True)
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
    '''
    drive_range = np.linspace(2.0,4.0,5)
    PL_range = np.linspace(np.pi/16,3*np.pi/8,5)
    

    parameter_grid_example = {
        'drive_right': drive_range,
        'PL': PL_range
    }
    '''

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_example,
        common_kwargs={'fast': True, 'headless': True},
    )
    '''
    parameter_grid_example_2 = {
        'drive_right': drive_range,
        'drive_left': drive_range
    }
    '''
    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_example_2,
        common_kwargs={'fast': True, 'headless': True},
    )'''

    plot = kwargs.pop('plot', False)
    if plot:
        metrics = []
        metrics_2 = []
        for drive_val in parameter_grid_example['drive_right']:
            for PL_val in parameter_grid_example['PL']:
                sim_result = BASE_PATH + \
                f'simulation_drive_right{drive_val:0.3f}_PL{PL_val:0.3f}.hdf5'
                v_fwd,_, cot = load_metrics_from_hdf5(sim_result)
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
                v_fwd,v_lat, _ = load_metrics_from_hdf5(sim_result)
                int_results = {
                    'drive right': drive_val_right,
                    'drive left': drive_val_left,
                    'forward_speed': v_fwd,
                    'lateral speed': v_lat
                }
                metrics_2.append(int_results)

        drive_vals = np.array([m['drive'] for m in metrics])
        PL_vals = np.array([m['phase lag'] for m in metrics])
        forward_speed_vals = np.array([m['forward_speed'] for m in metrics])
        cot_vals = np.array([m['CoT'] for m in metrics])

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

        plt.show()
        print(metrics_2)


if __name__ == '__main__':
    exercise2_2(plot=True)

