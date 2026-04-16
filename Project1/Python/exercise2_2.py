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
    ''' os.makedirs(PLOT_PATH, exist_ok=True)
    base_controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'freq': 1.5,
            'twl': 0.2,
            'amp': 1.0}}


    example_twl_range = np.linspace(0.2, 1.5, 10)
    example_amp_range = np.linspace(1.0, 4.0, 10)

    parameter_grid_example = {
        'twl': example_twl_range,
        'amp': example_amp_range,
    }

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid_example,
        common_kwargs={'fast': True, 'headless': True},
    )'''

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_2(plot=True)

