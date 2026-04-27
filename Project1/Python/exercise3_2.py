#!/usr/bin/env python3

import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    filter_signals,
)
from simulate import run_multiple

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
MAX_WORKERS = 16  # adjust based on your hardware capabilities


BASE_PATH = 'logs/exercise3_2/'
PLOT_PATH = 'results'

# CPG parameters
DRIVE_LEFT = 3
DRIVE_RIGHT = 3
DRIVE_LOW = 1
DRIVE_HIGH = 5
A_RATE = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP = np.ones(8) * 0.5
G_FREQ = np.ones(8) * 0.5
G_AMP = np.ones(8) * 0.25
PHASELAG =  np.ones(7)*np.pi * 2 / 8
COUPLING_WEIGHTS_ROSTRAL = 5
COUPLING_WEIGHTS_CAUDAL = 5
COUPLING_WEIGHTS_CONTRA = 10
# random init phases for 16 oscillators for 8 joints
INIT_PHASE = np.random.default_rng(
    seed=42).uniform(0.0, 2 * np.pi, size=16)

pylog.set_level('warning')
# pylog.set_level('critical') # suppress logging output in multi-processing

#added by Matt on the basis of ex 1.2
def get_metrics(w_ipsi):
    """Compute mechanical metrics for a single parameter set."""
    # Load HDF5
    sim_result = BASE_PATH + \
        f'simulation_w_ipsi{w_ipsi:0.3f}.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    transient_cutoff = 2000
    

    sim_times_steady = sim_times[transient_cutoff:]

    links_positions_steady = sensor_data_links[transient_cutoff:, :, 7:10]
    links_velocities_steady = sensor_data_links[transient_cutoff:, :, 14:17]
    joints_velocities_steady = sensor_data_joints[transient_cutoff:, :, 1]
    joints_torques_steady = sensor_data_joints[transient_cutoff:, :, 2]

    speed_forward, _ = compute_mechanical_speed(
        links_positions=links_positions_steady,
        links_velocities=links_velocities_steady,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=links_positions_steady,
        joints_torques=joints_torques_steady,
        joints_velocities=joints_velocities_steady,
    )

    # Load Controller
    controller_file = os.path.join(
        BASE_PATH,
        f"controller_w_ipsi{w_ipsi:0.3f}.pkl",
    )
    with open(controller_file, "rb") as f:
        controller_data = pickle.load(f)
    indices = controller_data["indices"]

    idx_L = indices['left_body_idx']
    idx_R = indices['right_body_idx']
    
    amp_idx_L = slice(idx_L.start + 16, idx_L.stop + 16, idx_L.step)
    amp_idx_R = slice(idx_R.start + 16, idx_R.stop + 16, idx_R.step)

    phases_left = controller_data["state"][:, idx_L]
    amps_left = controller_data["state"][:, amp_idx_L]
    
    phases_right = controller_data["state"][:, idx_R]
    amps_right = controller_data["state"][:, amp_idx_R]
    
    motor_left = amps_left * (1 + np.cos(phases_left))
    motor_right = amps_right * (1 + np.cos(phases_right))
    
    neural_signals = motor_left - motor_right
    
    neural_signals_steady = neural_signals[transient_cutoff:]
    sim_times_steady = sim_times[transient_cutoff:]
    
    neural_signals_smoothed = filter_signals(times=sim_times_steady, signals=neural_signals_steady)
    
    peak_frq, freq, peak_amp = compute_frequency_amplitude_fft(
        times=sim_times_steady,
        smooth_signals=neural_signals_smoothed,
    )
    return speed_forward, cot, peak_frq, peak_amp

def exercise3_2(**kwargs):
    """ex3.2 main"""
    os.makedirs(os.path.join(BASE_PATH, PLOT_PATH), exist_ok=True)
    #pylog.warning("TODO: 3.2 Explore the effect of stretch feedback on the metrics.")

    w_ipsi_range = np.linspace(-3.0, 17.0, 80)

    controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': DRIVE_LEFT,
            'drive_right': DRIVE_RIGHT,
            'd_low': DRIVE_LOW,
            'd_high': DRIVE_HIGH,
            'a_rate': A_RATE,
            'offset_freq': OFFSET_FREQ,
            'offset_amp': OFFSET_AMP,
            'G_freq': G_FREQ,
            'G_amp': G_AMP,
            'PL': PHASELAG,
            'coupling_weights_rostral': COUPLING_WEIGHTS_ROSTRAL,
            'coupling_weights_caudal': COUPLING_WEIGHTS_CAUDAL,
            'coupling_weights_contra': COUPLING_WEIGHTS_CONTRA,
            'init_phase': INIT_PHASE,
        },
    }
    '''run_multiple(
        max_workers=MAX_WORKERS,
        controller=controller,
        base_path=BASE_PATH,
        parameter_grid={'w_ipsi': w_ipsi_range},
        common_kwargs={
            'fast': True,
            'headless': True,
            'runtime_n_iterations': 20001,
            'runtime_buffer_size': 20001,
        },
    )'''

    metrics = []
    for w_ipsi_val in w_ipsi_range:
        v_fwd, cot, peak_freq, peak_amp = get_metrics(w_ipsi=w_ipsi_val)
        int_results = {
            'w_ipsi': w_ipsi_val,
            'forward_speed': v_fwd,
            'CoT': cot,
            'peak_frequency': peak_freq,
            'peak_amplitude': peak_amp
        }
        metrics.append(int_results)

    # ---- Plots: no saving, no 3D ----
    w_vals = np.array([m['w_ipsi'] for m in metrics])
    fwd_vals = np.array([m['forward_speed'] for m in metrics])
    cot_vals = np.array([m['CoT'] for m in metrics])
    freq_vals = np.array([m['peak_frequency'] for m in metrics])   # shape: (20, 8)
    amp_vals = np.array([m['peak_amplitude'] for m in metrics])    # shape: (20, 8)

    figs = []

    # 1) Forward speed (2D scatter)
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.scatter(w_vals, fwd_vals, s=45, alpha=0.9)
    ax1.set_title('Forward Speed vs w_ipsi')
    ax1.set_xlabel('w_ipsi [-]')
    ax1.set_ylabel('Forward speed [m/s]')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(BASE_PATH, PLOT_PATH, "forward_speed_vs_w_ipsi_3_2.png"), dpi=150)
    figs.append(fig1)

    # 2) CoT (2D scatter)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(w_vals, cot_vals, s=45, alpha=0.9, color='tab:orange')
    ax2.set_title('CoT vs w_ipsi')
    ax2.set_xlabel('w_ipsi [-]')
    ax2.set_ylabel('CoT [-]')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(BASE_PATH, PLOT_PATH, "cot_vs_w_ipsi_3_2.png"), dpi=150)
    figs.append(fig2)

    # 3) Peak frequency: single plot using mean across joints
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(1, 1, 1)

    # Robust to either shape (N, 8) or (N,)
    if freq_vals.ndim == 2:
        freq_mean = np.mean(freq_vals, axis=1)
    else:
        freq_mean = freq_vals

    ax3.scatter(w_vals, freq_mean, s=45, alpha=0.9, color='tab:green', label='Mean peak frequency')
    ax3.plot(w_vals, freq_mean, color='tab:green', alpha=0.6)
    ax3.set_title('Mean Peak Frequency vs w_ipsi')
    ax3.set_xlabel('w_ipsi [-]')
    ax3.set_ylabel('Mean peak frequency [Hz]')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(BASE_PATH, PLOT_PATH, "peak_frequency_vs_w_ipsi_3_2.png"), dpi=150)
    figs.append(fig3)

    # 4) Peak amplitude: combined in a single plot
    fig4 = plt.figure(figsize=(9, 6))
    ax4 = fig4.add_subplot(1, 1, 1)

    if amp_vals.ndim == 2:
        # One curve per channel
        for j in range(amp_vals.shape[1]):
            ax4.plot(
                w_vals, amp_vals[:, j],
                marker='o', markersize=3, linewidth=1.2, alpha=0.8,
                label=f'Channel {j}'
            )
        # Optional summary curve
        amp_mean = np.mean(amp_vals, axis=1)
        ax4.plot(
            w_vals, amp_mean,
            color='black', linewidth=2.2, alpha=0.9,
            label='Mean (all channels)'
        )
    else:
        # Fallback if amplitude is already 1D
        ax4.plot(
            w_vals, amp_vals,
            marker='o', markersize=4, linewidth=1.4, alpha=0.9,
            color='tab:red', label='Peak amplitude'
        )

    ax4.set_title('Peak Amplitude vs w_ipsi (all channels)')
    ax4.set_xlabel('w_ipsi [-]')
    ax4.set_ylabel('Peak amplitude [a.u.]')
    ax4.grid(True, alpha=0.3)
    ax4.legend(ncol=2, fontsize=8)
    fig4.tight_layout()
    fig4.savefig(os.path.join(BASE_PATH, PLOT_PATH, "peak_amplitude_vs_w_ipsi_3_2.png"), dpi=150)
    figs.append(fig4)

    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()
    else:
        for f in figs:
            plt.close(f)

if __name__ == '__main__':
    exercise3_2(plot=True)

