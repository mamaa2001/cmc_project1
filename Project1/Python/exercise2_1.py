#!/usr/bin/env python3

import time
import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt


from farms_core import pylog
from farms_core.utils.profile import profile

from simulate import runsim
from cmc_controllers.metrics import filter_signals

BASE_PATH = 'logs/exercise2_1/'
PLOT_PATH = 'results'


def post_processing(base_path):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        state_data = f['FARMSLISTanimats']['0']['state'][:] #rajout pour les plots
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    # Load Controller
    with open(base_path + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)
    print("controleur data keys:", controller_data.keys())

    # pour l'instant c'est guez de ouf

    # θ et r directement depuis HDF5
    theta_hist = state_data[:, :16]   # colonnes 0-15
    r_hist     = state_data[:, 16:32] # colonnes 16-31

    # Filtre 5s
    mask = sim_times <= 5.0
    t = sim_times[mask]

    # Plot θ
    plt.figure()
    for i in range(16):
        plt.plot(t, theta_hist[mask, i], label=f"θ_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Phase θ")
    plt.title("Time evolution of phases (θ)"); plt.legend(); plt.grid()

    # Plot r
    plt.figure()
    for i in range(16):
        plt.plot(t, r_hist[mask, i], label=f"r_{i}")
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude r")
    plt.title("Time evolution of amplitudes (r)"); plt.legend(); plt.grid()

    #########################################################################

def main(**kwargs):
    """Run exercise 2.1 simulation and post-processing pipeline."""
    os.makedirs(PLOT_PATH, exist_ok=True)
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
                size=16)}}

    tic = time.time()
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording='exercise2_1.mp4',
    )
    post_processing(BASE_PATH)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


def exercise2_1(**kwargs):
    """ex2.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()

if __name__ == '__main__':
    exercise2_1(plot=True)

