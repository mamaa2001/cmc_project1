"""Exercise 4: Transitions between swimming and walking"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters
#import farms_pylog as pylog

def exercise_4a_transition(timestep):

    # --- Walk to swim ---
    os.makedirs('./logs/ex4_walk2swim/', exist_ok=True)
    simulation(
        sim_parameters=SimulationParameters(
            duration=90,
            timestep=timestep,
            spawn_position=[-0.75, 0, 0.1],
            spawn_orientation=[0, 0, np.pi],
            drive=2.0,
            update_drive=True,
        ),
        arena='amphibious',
        fast=True,
        output='logs/ex4_walk2swim/sim_0',
        record=True,
        record_path='logs/ex4_walk2swim/video_walk2swim.mp4',
        verbose=True,
    )

    # --- Swim to walk ---
    os.makedirs('./logs/ex4_swim2walk/', exist_ok=True)
    simulation(
        sim_parameters=SimulationParameters(
            duration=90,
            timestep=timestep,
            spawn_position=[1.5, 0, 0.1],
            spawn_orientation=[0, 0, 0],
            drive=4.0,
            update_drive=True,
        ),
        arena='amphibious',
        fast=True,
        output='logs/ex4_swim2walk/sim_0',
        record=True,
        record_path='logs/ex4_swim2walk/video_swim2walk.mp4',
        verbose=True,
    )
    return


if __name__ == '__main__':
    exercise_4a_transition(timestep=5e-3)

