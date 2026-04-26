#!/usr/bin/env python3

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
)
from simulate import runsim, run_multiple

pylog.set_level('warning')
# pylog.set_level('critical') # suppress logging output in multi-processing

BASE_PATH = 'logs/exercise3_3/'
PLOT_PATH = 'results'

# CPG parameters
DRIVE_LEFT = 3
DRIVE_RIGHT = 3
DRIVE_LOW = 1
DRIVE_HIGH = 5
A_RATE = np.ones(8) * 3
OFFSET_FREQ = np.ones(8) * 1
OFFSET_AMP = np.ones(8) * 0
G_FREQ = np.ones(8) * 0.5
G_AMP = np.ones(8) * 0.25
PHASELAG = np.ones(7) * np.pi * 2 / 8
COUPLING_WEIGHTS_ROSTRAL = 5
COUPLING_WEIGHTS_CAUDAL = 5
COUPLING_WEIGHTS_CONTRA = 10
# random init phases for 16 oscillators for 8 joints
INIT_PHASE = np.random.default_rng(
    seed=42).uniform(0.0, 2 * np.pi, size=16)
W_IPSI = 10.0

# disruption propabilities
DISRUPTION_P_SENSORS = 0.2
DISRUPTION_P_COUPLINGS = 0.2
RANDOM_SEED = 42
MAX_WORKERS = 8

def load_sim_data(hdf5_path, skip_start=500):
    """Load simulation sensor data and slice out initial transient."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f["times"][:]
        sensor_links = f["FARMSLISTanimats"]["0"]["sensors"]["links"]["array"][:]
        sensor_joints = f["FARMSLISTanimats"]["0"]["sensors"]["joints"]["array"][:]

    sim_times = sim_times[skip_start:]
    sensor_links_pos = sensor_links[skip_start:, :, 7:10]
    sensor_joints_pos = sensor_joints[skip_start:, :, 0]

    return sim_times, sensor_links_pos, sensor_joints_pos


def exercise3_3(**kwargs):
    """ex3.3 main"""
    #pylog.warning("TODO: 3.3 Implement neural disruptions and compare with no disruption.")

    ##################### code estelle #######################
    # --- Baseline: no disruption ---
    baseline_config = dict(
        drive_left=DRIVE_LEFT, drive_right=DRIVE_RIGHT,
        d_low=DRIVE_LOW, d_high=DRIVE_HIGH,
        a_rate=A_RATE, offset_freq=OFFSET_FREQ, offset_amp=OFFSET_AMP,
        G_freq=G_FREQ, G_amp=G_AMP, PL=PHASELAG,
        coupling_weights_rostral=COUPLING_WEIGHTS_ROSTRAL,
        coupling_weights_caudal=COUPLING_WEIGHTS_CAUDAL,
        coupling_weights_contra=COUPLING_WEIGHTS_CONTRA,
        init_phase=INIT_PHASE,
        w_ipsi=W_IPSI,
        disruption_p_sensors=0.0,
        disruption_p_couplings=0.0,
        random_seed=RANDOM_SEED,
    )
    runsim(baseline_config, log_path=BASE_PATH+'baseline/', n_iterations=10001)

    # --- Mixed disruption demo (20%/20%) ---
    mixed_config = {**baseline_config,
        'disruption_p_sensors': DISRUPTION_P_SENSORS,
        'disruption_p_couplings': DISRUPTION_P_COUPLINGS,
    }
    runsim(mixed_config, log_path=BASE_PATH+'mixed_demo/', n_iterations=10001)

    # Building of all configurations (Two setups × three disruption types × five probabilities = 30 configs:)
    probs = np.linspace(0, 0.15, 5)
    configs = []
    log_paths = []

    for setup_name, w_ipsi, rostral, caudal in [
        ('combined',  W_IPSI, COUPLING_WEIGHTS_ROSTRAL, COUPLING_WEIGHTS_CAUDAL),
        ('decoupled', W_IPSI, 0.0, 0.0),  # ipsilateral coupling removed
    ]:
        for disruption_type, p_sensors, p_couplings in [
            ('muted_sensors',      probs, np.zeros(5)),
            ('removed_couplings',  np.zeros(5), probs),
            ('mixed',              probs, probs),
        ]:
            for p_s, p_c in zip(p_sensors, p_couplings):
                cfg = {**baseline_config,
                    'coupling_weights_rostral': rostral,
                    'coupling_weights_caudal': caudal,
                    'w_ipsi': w_ipsi,
                    'disruption_p_sensors': p_s,
                    'disruption_p_couplings': p_c,
                }
                configs.append(cfg)
                log_paths.append(
                    f"{BASE_PATH}{setup_name}/{disruption_type}/p{p_s:.3f}_{p_c:.3f}/"
                )

    run_multiple(configs, log_paths=log_paths, n_iterations=10001, max_workers=MAX_WORKERS)

    # Loading results and computing metrics
    speeds = np.zeros((2, 3, 5))   # [setup, disruption_type, prob]
    cots   = np.zeros((2, 3, 5))

    for i_setup, setup_name in enumerate(['combined', 'decoupled']):
        for i_type, disruption_type in enumerate(['muted_sensors', 'removed_couplings', 'mixed']):
            for i_p, (p_s, p_c) in enumerate(zip(p_sensors_list[i_type], p_couplings_list[i_type])):
                path = f"{BASE_PATH}{setup_name}/{disruption_type}/p{p_s:.3f}_{p_c:.3f}/simulation.hdf5"
                sim_times, sensor_links_pos, sensor_joints_pos = load_sim_data(path)
                
                speed, _ = compute_mechanical_speed(sim_times, sensor_links_pos)
                _, cot = compute_mechanical_energy_and_cot(sim_times, sensor_joints_pos, sensor_links_pos)
                
                speeds[i_setup, i_type, i_p] = speed
                cots[i_setup, i_type, i_p] = cot

        plot = kwargs.pop('plot', False)
        if plot:
            plt.show()

    # Plotting 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ['Muted Sensors', 'Removed Couplings', 'Mixed']

    for i_setup, setup_name in enumerate(['Combined', 'Decoupled']):
        for i_type, label in enumerate(labels):
            axes[0, i_setup].plot(probs*100, speeds[i_setup, i_type], label=label)
            axes[1, i_setup].plot(probs*100, cots[i_setup, i_type],   label=label)

        axes[0, i_setup].set_title(f'{setup_name} — Speed')
        axes[1, i_setup].set_title(f'{setup_name} — CoT')
        for ax in axes[:, i_setup]:
            ax.set_xlabel('Disruption probability (%)')
            ax.legend()

    axes[0, 0].set_ylabel('Forward speed (m/s)')
    axes[1, 0].set_ylabel('CoT (J/m)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, PLOT_PATH, 'disruption_ablation.png'))

if __name__ == '__main__':
    exercise3_3(plot=True)

