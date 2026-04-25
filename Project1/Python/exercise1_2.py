#!/usr/bin/env python3
"""Run parameter sweeps for exercise 1.2 and plot metric heatmaps."""

import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

from farms_core import pylog

#pour les courbes
from exercise1_1 import post_processing

from simulate import run_multiple
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

# Multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MAX_WORKERS = 8  # adjust based on your hardware capabilities

pylog.set_level('critical')

BASE_PATH = 'logs/exercise1_2/'
PLOT_PATH = 'results'
RECORDING = None  # disable recording for parallel runs


def get_metrics(twl, amp):
    """Compute mechanical metrics for a single parameter set."""
    # Load HDF5
    sim_result = BASE_PATH + \
        f'simulation_twl{twl:0.3f}_amp{amp:0.3f}.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    sensor_data_links_positions = sensor_data_links[:, :, 7:10]

    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, _ = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    # Load Controller
    controller_file = os.path.join(
        BASE_PATH,
        f"controller_twl{twl:0.3f}_amp{amp:0.3f}.pkl",
    )
    with open(controller_file, "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)
    signal_freqs, _, _ = compute_frequency_amplitude_fft(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
    )
    inds_couples = [[i, i + 1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    _, ipls_mean = compute_neural_phase_lags(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
        freqs=signal_freqs,
        inds_couples=inds_couples,
    )

    return speed_forward, cot, float(ipls_mean)


def exercise1_2(**kwargs):
    """ex1.2 main"""
    os.makedirs(PLOT_PATH, exist_ok=True)
    base_controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {
            'freq': 1.5,
            'twl': 0.2,
            'amp': 1.0}}
    pylog.warning("TODO: 1.2 Adapt the parameter space according to needs.")
    # Hint: You don't need to test all combinations of parameters with complexity of O(n^3)
    # You can replace range with list of length 1 to keep some parameters fixed
    # while testing others O(n^2) or O(n)

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
    )

    pylog.warning("TODO: 1.3 Analyze the results of multiple simulations")
    
    #To displya the metrics
    metrics = []
    for twl_val in parameter_grid_example['twl']:
        for amp_val in parameter_grid_example['amp']:
            v_fwd, cot, mean_ipls = get_metrics(twl=twl_val, amp=amp_val)
            int_results = {
                'twl': twl_val,
                'amp': amp_val,
                'forward_speed': v_fwd,
                'CoT': cot,
                'average_ipls': mean_ipls
            }
            metrics.append(int_results)

    forward_speed_vals = np.array([m['forward_speed'] for m in metrics])
    cot_vals = np.array([m['CoT'] for m in metrics])
    ipls_vals = np.array([m['average_ipls'] for m in metrics])

    # Reshape flat metric arrays into 2D grids: rows=twl, cols=amp
    n_twl = len(parameter_grid_example['twl'])
    n_amp = len(parameter_grid_example['amp'])

    forward_speed_grid = forward_speed_vals.reshape(n_twl, n_amp)
    cot_grid = cot_vals.reshape(n_twl, n_amp)
    ipls_grid = ipls_vals.reshape(n_twl, n_amp)


    # Separate annotated heatmaps (one figure per metric)
    amp_axis = parameter_grid_example['amp']
    twl_axis = parameter_grid_example['twl']

    # Use one common colormap for all plots
    shared_cmap = 'viridis'

    def plot_annotated_heatmap(grid, title, cmap, cbar_label, value_fmt=".2f"):
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap)

        ax.set_title(title)
        ax.set_xlabel('A [-]')
        ax.set_ylabel('TWL [Body length]')

        # Show actual parameter values on axes
        ax.set_xticks(np.arange(len(amp_axis)))
        ax.set_xticklabels([f"{v:.2f}" for v in amp_axis], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(twl_axis)))
        ax.set_yticklabels([f"{v:.2f}" for v in twl_axis])

        # Annotate each cell with its value
        norm = colors.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                txt_color = 'white' if norm(val) < 0.6 else 'black'
                ax.text(j, i, format(val, value_fmt),
                        ha='center', va='center', color=txt_color, fontsize=8)

        # Frame lowest and highest values
        min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
        max_idx = np.unravel_index(np.nanargmax(grid), grid.shape)

        ax.add_patch(Rectangle(
            (min_idx[1] - 0.5, min_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=2.5
        ))
        ax.add_patch(Rectangle(
            (max_idx[1] - 0.5, max_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor='red', linewidth=2.5
        ))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    plot_annotated_heatmap(
        forward_speed_grid, 'Forward speed at f = 2 Hz', shared_cmap, 'Forward speed [m/s]', value_fmt=".3f"
    )
    plot_annotated_heatmap(
        cot_grid, 'CoT at f = 2 Hz', shared_cmap, 'CoT [J/m]', value_fmt=".3f"
    )
    plot_annotated_heatmap(
        ipls_grid,
        r'Average $IPL_{neur}$ at f = 2 Hz',
        shared_cmap,
        r'Average $IPL_{neur}$ [rad]',
        value_fmt=".3f"
    )

    plt.show()

if __name__ == '__main__':
    exercise1_2(plot=True)

