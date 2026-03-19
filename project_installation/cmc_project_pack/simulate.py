#!/usr/bin/env python3

import os
from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.extensions.extensions import import_item
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import (
    setup_from_clargs,
    run_simulation,
)

# Multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
MAX_WORKERS = 4

# DEBUG
# pylog.set_level('critical')

base_path = 'logs/'
recording = 'test.mp4'

def simulate(**kwargs):
    """Main Simulation function"""
    # Fix CWD to project root for relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    os.makedirs(base_path, exist_ok=True)
    arena          = kwargs.pop('arena', 'ground')
    headless       = kwargs.pop('headless', False)
    fast           = kwargs.pop('fast', False)
    hdf5_name      = kwargs.pop('hdf5_name', 'simulation.hdf5')
    recording_file = kwargs.pop('recording', recording)

    # Setup
    pylog.info('Loading options from clargs')
    clargs = sim_parse_args()
    if arena=='ground':
        clargs.experiment_config = os.path.join(os.path.dirname(__file__), "configs", "experiment_ground.yaml")
    elif arena=='water':
        clargs.experiment_config = os.path.join(os.path.dirname(__file__), "configs", "experiment_water.yaml")
    else:
        raise ValueError(f"Unknown arena: {arena}")
    _, exp_options, simulator = setup_from_clargs(clargs=clargs)

    # Controller
    controller = {'loader': 'cmc_controllers.dummy_controller.DummyController', 'config': {}}
    exp_options.animats[0].extensions.insert(0, controller)

    exp_options.simulation.runtime.headless = headless
    exp_options.simulation.runtime.fast = fast

    # Logger
    for ext in exp_options.simulation.extensions:
        if ext.loader == 'farms_core.simulation.extensions.ExperimentLogger':
            ext['config']['log_path'] = base_path
            ext['config']['log_name'] = hdf5_name
        if ext.loader == 'farms_mujoco.sensors.camera.CameraRecording':
            if not recording_file:
                exp_options.simulation.extensions.remove(ext)
            else:
                ext['config']['path'] = base_path + recording_file

    # Data
    experiment_data_loader = import_item(exp_options.loaders.experiment_data)
    experiment_data = experiment_data_loader.from_options(exp_options)

    # Simulation
    pylog.info('Creating simulation environment')
    sim = run_simulation(
        experiment_data=experiment_data,
        experiment_options=exp_options,
        simulator=simulator,
    )


def example_single(**kwargs):
    profile(function=simulate, profile_filename='',
            arena=kwargs.pop('arena', 'ground'),
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)

