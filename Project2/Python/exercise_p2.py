"""[Project1] Exercise 2: Swimming & Walking with Salamander Robot"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_walk(timestep, n_simulations=1):
    "[Project 1] Q2 Walking with an increasing (ramp) drive"
    parameter_set = [
        SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi / 2],
            drive=2,
        )
    ]

    os.makedirs('./logs/ex_2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        sim, data = simulation(
            sim_parameters=sim_parameters,
            arena='land',
            output=f'logs/example/sim_{simulation_i}',
            record=False,
            record_path=f"logs/example/video_{simulation_i}.mp4",
            verbose=True,
        )
    return


def exercise_ramp_swim(timestep):
    "[Project 1] Q2 Swimming with a linear drive ramp from 1 to 5 over 40s"
    os.makedirs('./logs/ex_ramp_swim/', exist_ok=True)
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=1.0,
        drive_ramp=True,
        drive_ramp_start=1.0,
        drive_ramp_end=5.0,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='water',
        output='logs/ex_ramp_swim/sim_0',
        record=False,
        record_path="logs/ex_ramp_swim/video_ramp_swim.mp4",
        verbose=True,
    )
    return


def exercise_ramp_walk(timestep):
    "[Project 1] Q2 Walking with a linear drive ramp from 1 to 5 over 40s"
    os.makedirs('./logs/ex_ramp_walk/', exist_ok=True)
    sim_parameters = SimulationParameters(
        duration=40,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=1.0,
        drive_ramp=True,
        drive_ramp_start=1.0,
        drive_ramp_end=5.0,
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        output='logs/ex_ramp_walk/sim_0',
        record=False,
        record_path="logs/ex_ramp_walk/video_ramp_walk.mp4",
        verbose=True,
    )
    return


if __name__ == '__main__':
    exercise_walk(timestep=5e-3)
    exercise_ramp_walk(timestep=5e-3)