"""[Project1] Exercise 1: Implement & run network without MuJoCo"""

import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from salamandra_simulation.data import SalamandraState
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from simulation_parameters import SimulationParameters
from network import SalamandraNetwork


@dataclass
class DataState:
    state: SalamandraState


def run_network(duration, update=False, drive=0, timestep=1e-2):
    """ Run network without MuJoCo and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        True: use the prescribed drive parameter, False: update the drive during the simulation
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)

    if np.isscalar(drive):
        drive_vec = np.full(n_iterations, drive)
    else:
        drive_vec = np.asarray(drive)

    sim_parameters = SimulationParameters(
        drive=drive,
        amplitude_gradient=None,
        phase_lag_body=None,
        test_value = 10,
        # Feel free to include parameters
    )
    #pylog.warning(
    #    'Modify the scalar drive to be a vector of length n_iterations. By doing so the drive will be modified to be drive[i] at each time step i.')
    state = SalamandraState.salamandra_robot(n_iterations, n_oscillators=32)
    network = SalamandraNetwork(
        sim_parameters,
        n_iterations,
        DataState(
            state=state))
    osc_left = np.arange(0, 16, 2)
    osc_right = np.arange(1, 16, 2)
    osc_legs = np.arange(16, 32)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs

    # comment below pass to run file
    #pylog.warning('Remove the pass to run your code!!')
    #pass

    pylog.warning(
        'Implement plots here, try to plot the various logged data to check the implementation')
    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive=drive_vec[i + 1],
                    amplitude_gradient=None,
                    phase_lag_body=None,
                )
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs
    toc = time.time()

    # Network performance
    pylog.info('Time to run simulation for {} steps: {} [s]'.format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results
    #pylog.warning('Implement plots')


    # ── Plots (mirrors Figures 4 & 5 from Ijspeert 2007) ────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Network dynamics (drive ramp 0→6)')

    # Panel A: body oscillator outputs  x_i = r_i(1 + cos(φ_i))
    body_output = amplitudes_log[:, :16] * (1 + np.cos(phases_log[:, :16]))
    for idx in osc_left:   # left side only, less clutter
        axes[0].plot(times, body_output[:, idx], lw=0.8)
    axes[0].set_ylabel('x Body (left)')
    axes[0].set_title('A – Body oscillator outputs')

    # Panel B: limb oscillator outputs
    limb_output = amplitudes_log[:, 16:] * (1 + np.cos(phases_log[:, 16:]))
    for idx in range(0, 16, 4):   # one oscillator per limb
        axes[1].plot(times, limb_output[:, idx], lw=0.8, label=f'limb {idx//4}')
    axes[1].set_ylabel('x Limb')
    axes[1].set_title('B – Limb oscillator outputs')
    axes[1].legend(fontsize=7, loc='upper left')

    # Panel C: instantaneous frequencies
    axes[2].plot(times, freqs_log[:, 0], label='Body', lw=1.2)
    axes[2].plot(times, freqs_log[:, 16], label='Limb', lw=1.2, linestyle='--')
    axes[2].set_ylabel('Freq [Hz]')
    axes[2].set_title('C – Frequencies')
    axes[2].legend(fontsize=8)

    # Panel D: drive ramp
    axes[3].plot(times, drive_vec, color='red', lw=1.2)
    axes[3].axhline(1.0, color='gray', linestyle=':', lw=0.8, label='walk threshold')
    axes[3].axhline(3.0, color='blue', linestyle=':', lw=0.8, label='swim threshold')
    axes[3].axhline(5.0, color='black', linestyle=':', lw=0.8, label='saturation')
    axes[3].set_ylabel('Drive d')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_title('D – Drive')
    axes[3].legend(fontsize=7)

    plt.tight_layout()

    return


def exercise_1a_networks(plot, timestep=1e-2):
    """[Project 1] Exercise 1: """

    duration = 40.0
    times = np.arange(0, duration, timestep)
    drive_ramp = np.linspace(0, 6, len(times))

    run_network(duration=duration, update=True, drive=drive_ramp, timestep=timestep)

    
    #run_network(duration=5)

    # Show plots
    if True:
        if plot:
            plt.show()
        else:
            save_figures()
        return


if __name__ == '__main__':
    exercise_1a_networks(plot=not save_plots())

