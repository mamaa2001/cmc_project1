#!/usr/bin/env python3

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

    pylog.warning("TODO: 3.1 Simulate with and without sensory feedback")

    pylog.warning("TODO: 3.1 Compare the performance")


    runsim(
        controller=controller,
        base_path=BASE_PATH,
        w_ipsi=w_ipsi,
        recording='animation3_1_with_sf.mp4',
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
        recording='animation3_1_without_sf.mp4',
        hdf5_name='simulation_without_sf.hdf5',
        controller_name='controller_without_sf.pkl',
        runtime_n_iterations=5001,
        runtime_buffer_size=5001,
        fast=fast,
        headless=headless,
    )

    #plot_oscillator_states()
  
def exercise3_1(**kwargs):
    """ex3.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()

"""
def plot_oscillator_states():
    # Plot oscillator states (theta and r) for w_ipsi=3 and w_ipsi=0

    with open(os.path.join(BASE_PATH, 'controller_with_sf.pkl'), 'rb') as f:
        controller_with = pickle.load(f)
    with open(os.path.join(BASE_PATH, 'controller_without_sf.pkl'), 'rb') as f:
        controller_without = pickle.load(f)

    # Extract states directly from the dict
    state_with = controller_with['state']       # shape (n_iterations, 3*n_oscillators)
    state_without = controller_without['state']

    print("state_with shape:", state_with.shape)
    print("non-zero rows:", np.sum(np.any(state_with != 0, axis=1)))
    print("timestep would be:", 5.0 / state_with.shape[0])
    return

    n_oscillators = 16
    n_iterations = state_with.shape[0]
    timestep = 0.001  # adjust if different
    time = np.arange(n_iterations) * timestep

    # Extract theta and r
    theta_with    = state_with[:, :n_oscillators]
    r_with        = state_with[:, n_oscillators:2*n_oscillators]
    theta_without = state_without[:, :n_oscillators]
    r_without     = state_without[:, n_oscillators:2*n_oscillators]

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

"""

if __name__ == '__main__':
    exercise3_1(plot=True)

