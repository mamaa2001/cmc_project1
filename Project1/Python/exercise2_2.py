#!/usr/bin/env python3
"""Run exercise 2.2 parameter sweeps and generate heatmaps/trajectory plots."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

from farms_core import pylog

from cmc_controllers.metrics import (
    get_filtered_signals,
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
    CoM_traj_filtered = get_filtered_signals(CoM_traj, signal_dt=timestep, fcut_lp=0.5)
    
    curvature_mean = compute_trajectory_curvature(trajectory=CoM_traj_filtered, timestep=timestep)

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

        # ---- Heatmaps (same palette + value in each square) ----
        shared_cmap = "viridis"

        def plot_annotated_heatmap(grid, x_vals, y_vals, title, xlabel, ylabel, cbar_label, fmt=".3f", frame_abs_min=False):
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap=shared_cmap)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_xticks(np.arange(len(x_vals)))
            ax.set_xticklabels([f"{v:.3f}" for v in x_vals], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(y_vals)))
            ax.set_yticklabels([f"{v:.3f}" for v in y_vals])

            norm = colors.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]
                    txt_color = "white" if norm(val) < 0.6 else "black"
                    ax.text(j, i, format(val, fmt), ha="center", va="center", color=txt_color, fontsize=8)

            # Frame lowest and highest values
            min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
            max_idx = np.unravel_index(np.nanargmax(grid), grid.shape)

            ax.add_patch(Rectangle(
                (min_idx[1] - 0.5, min_idx[0] - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=2.5
            ))
            ax.add_patch(Rectangle(
                (max_idx[1] - 0.5, max_idx[0] - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=2.5
            ))

            # Frame lowest absolute value if requested
            if frame_abs_min:
                abs_min_idx = np.unravel_index(np.nanargmin(np.abs(grid)), grid.shape)
                ax.add_patch(Rectangle(
                    (abs_min_idx[1] - 0.5, abs_min_idx[0] - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.5
                ))

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)

        # Grid 1: rows=drive, cols=PL
        unique_drive = np.sort(np.unique(drive_vals))
        unique_PL = np.sort(np.unique(PL_vals))
        forward_speed_grid = forward_speed_vals.reshape(len(unique_drive), len(unique_PL))
        cot_grid = cot_vals.reshape(len(unique_drive), len(unique_PL))

        plot_annotated_heatmap(
            forward_speed_grid,
            x_vals=unique_PL,
            y_vals=unique_drive,
            title="Forward speed [m/s]",
            xlabel="phase lag [rad]",
            ylabel="drive [-]",
            cbar_label="forward speed [m/s]",
            fmt=".3f",
        )

        plot_annotated_heatmap(
            cot_grid,
            x_vals=unique_PL,
            y_vals=unique_drive,
            title="Cost of Transport [J/m]",
            xlabel="phase lag [rad]",
            ylabel="drive [-]",
            cbar_label="CoT [J/m]",
            fmt=".3f",
        )

        # Grid 2: rows=drive_right, cols=drive_left
        unique_drive_right = np.sort(np.unique(drive_right_vals))
        unique_drive_left = np.sort(np.unique(drive_left_vals))
        mean_curvatures_vals = np.array([m['curvature mean'] for m in metrics_2])
        mean_curvature_grid = mean_curvatures_vals.reshape(len(unique_drive_right), len(unique_drive_left))

        plot_annotated_heatmap(
            mean_curvature_grid,
            x_vals=unique_drive_left,
            y_vals=unique_drive_right,
            title=r"Mean curvature [$m^{-1}$]",
            xlabel="drive left [-]",
            ylabel="drive right [-]",
            cbar_label=r"Mean curvature [$m^{-1}$]",
            fmt=".3f",
            frame_abs_min=True,
        )

        # Grid of CoM trajectories (one subplot per drive_right/drive_left pair)
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
                ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, label=f"drive left={dl:.2f} [-]")

            ax.set_title(f"drive right = {dr:.2f} [-]", fontsize=9)
            ax.set_xlabel("CoM x [m]")
            ax.set_ylabel("CoM y [m]")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=7, ncol=1)

        # Hide any unused subplot if unique_drive_right has fewer than 9 values
        for j in range(len(unique_drive_right), 9):
            axes[j].axis("off")

        fig3.suptitle("CoM trajectories grouped by drive right (x,y in [m])", y=0.98)
        plt.tight_layout()

        plt.show()


if __name__ == '__main__':
    exercise2_2(plot=True)

