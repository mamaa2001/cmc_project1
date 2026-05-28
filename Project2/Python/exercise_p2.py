"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_walk(timestep, n_simulations = 1):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=50,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            # Orientation in Euler angles [rad]
            spawn_orientation=[0, 0, np.pi/2],
            drive=drive,  # An example of parameter part of the grid search
            amplitudes=[1, 2, 3],  # Just an example
            phase_lag_body=None,  # or np.zeros(n_joints) for example
            turn=0,  # Another example
            position_body_gain = 1.75,
            position_limb_gain = 1.5,
            # ...
        )
        for drive in np.linspace(2, 3, n_simulations)
        # for amplitudes in ...
        # for ...
    ]

    # Run simulations
    os.makedirs('./logs/ex_2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='land',  # Can also be 'water', give it a try!
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            output=f'logs/example/sim_{simulation_i}',
            record=False,  # Record video
            # video savging path
            record_path=f"logs/example/video_{simulation_i}.mp4",
            verbose=True,
        )
    pass
    return


def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    pass
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    # Use exercise_example.py for reference
    pass
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    #exercise_ramp_swim(timestep=5e-3)
    #exercise_ramp_walk(timestep=5e-3)

