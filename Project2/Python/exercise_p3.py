"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
import matplotlib.pyplot as plt
from salamandra_simulation.data import SalamandraData


def compute_phase_lags_from_phases(phases_array, body_left_idx):
    """
    Compute inter-joint phase lags directly from CPG oscillator phases.
    Uses the second half (steady state) and computes mean phase difference.
    
    phases_array : shape [n_iter, 32]
    body_left_idx: e.g. [0,2,4,6,8,10,12,14]
    """
    # Use second half for steady state
    half = phases_array.shape[0] // 2
    phases_body = phases_array[half:, body_left_idx]  # shape [n/2, 8]
    
    lags = []
    for k in range(phases_body.shape[1] - 1):
        # Phase difference between consecutive joints, wrapped to [-pi, pi]
        diff = phases_body[:, k] - phases_body[:, k+1]
        diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))  # wrap
        lags.append(np.mean(diff_wrapped))
    
    return np.array(lags)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: plot spine phase lags and joint angles
# ─────────────────────────────────────────────────────────────────────────────
def plot_spine_analysis(times, joint_positions, lags_rad, drive_value, mode_label):
    """
    Two-panel figure:
      Top   : joint angle time series for all 8 body joints
      Bottom: phase lag between consecutive joints (bar chart)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f'Spine analysis — {mode_label}  (drive={drive_value})', fontsize=13)

    # Panel 1: joint angle traces (left side: joints 0,2,4,6)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, 8))
    for j in range(8):
        ax.plot(times, joint_positions[:, j], color=colors[j],
                lw=0.9, label=f'Joint {j}')
    ax.set_ylabel('Joint angle [rad]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Body joint angles (head→tail)')
    ax.legend(fontsize=7, ncol=4, loc='upper right')

    # Panel 2: phase lags between consecutive joints
    ax2 = axes[1]
    joint_pairs = [f'{k}→{k+1}' for k in range(len(lags_rad))]
    bar_colors = ['steelblue' if l >= 0 else 'tomato' for l in lags_rad]
    ax2.bar(joint_pairs, lags_rad, color=bar_colors)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axhline(2 * np.pi / 8, color='gray', linestyle='--', lw=1,
                label='Ideal swim lag (2π/8)')
    ax2.set_ylabel('Phase lag [rad]')
    ax2.set_xlabel('Joint pair')
    ax2.set_title('Inter-joint phase lags (positive = head leads tail)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs('./logs/ex3_1/', exist_ok=True)
    plt.savefig(f'./logs/ex3_1/spine_analysis_{mode_label}.png', dpi=150)
    pylog.info(f'Saved figure: logs/ex3_1/spine_analysis_{mode_label}.png')


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 — Analyze spine movement during walking
# ─────────────────────────────────────────────────────────────────────────────
def exercise_3_1_spine_analysis(timestep):
    """
    Exercise 3.1 — Spine movement analysis during walking.

    Runs two simulations:
      (a) Walking regime  (drive = 2.0, limbs active)
      (b) Swimming regime (drive = 4.0, limbs silent)

    For each, we extract:
      • Body joint angle time series
      • Inter-joint phase lags (via cross-correlation)
      • Dominant oscillation frequency

    This lets us answer:
      Q1: What are the phase lags along the spine during walking?
      Q2: What gait does the robot employ?
      Q3: How does spine movement differ between walking and swimming?
    """

    configs = [
        # (drive, arena,  label)
        (2.0, 'land',  'walking'),
        (4.0, 'water', 'swimming'),
    ]

    results = {}

    for drive_val, arena, label in configs:
        pylog.info(f'Running {label} simulation (drive={drive_val}, arena={arena})')

        sim_parameters = SimulationParameters(
            duration=10,
            timestep=timestep,
            drive=drive_val,
            amplitude_gradient=None,
            phase_lag_body=None,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
        )

        os.makedirs(f'./logs/ex3_1/{label}/', exist_ok=True)

        sim, sim_data = simulation(          # ← unpack the tuple
            sim_parameters=sim_parameters,
            arena=arena,
            fast=True,
            headless=True,
            output=f'./logs/ex3_1/{label}/',
        )
        #print(type(sim_data.sensors.joints))
        #print(dir(sim_data.sensors.joints))
        """pos_test = np.array(sim_data.sensors.joints.positions_all())
        cmd_test = np.array(sim_data.sensors.joints.cmd_positions())
        print(f"positions_all shape: {pos_test.shape}, max: {pos_test.max():.4f}, min: {pos_test.min():.4f}")
        print(f"cmd_positions shape: {cmd_test.shape}, max: {cmd_test.max():.4f}, min: {cmd_test.min():.4f}")
"""
        state_array = np.array(sim_data.state.array)
        print(f"State shape: {state_array.shape}")
        print(f"Phases max: {state_array[:, :32].max():.4f}")
        print(f"Amplitudes max: {state_array[:, 32:64].max():.4f}")

        # Also check if the network even oscillated
        phases = state_array[:, :32]
        print(f"Phase range joint 0: {phases[:, 0].min():.3f} to {phases[:, 0].max():.3f}")
        # ── Extract joint positions ──────────────────────────────────────────
        # sim_data.sensors.joints.positions_all() → shape [n_iter, n_joints]
        # First 8 columns = body joints (head→tail)
        #all_joint_pos = np.array(sim_data.sensors.joints.positions_all())
        #print(dir(sim_data.sensors.joints))
        # Replace the all_joint_pos line with this:
        state_array = np.array(sim_data.state.array)

        # Phases for left body oscillators (indices 0,2,4,6,8,10,12,14)
        # Amplitudes for same
        phases_all = state_array[:, :32]
        amps_all = state_array[:, 32:64]

        # Muscle output: M_i = r_i * (1 + cos(phi_i))  — Eq.3 from the assignment
        # For body joints, use left-side oscillators (even indices 0..14)
        body_left_idx = np.arange(0, 16, 2)   # [0,2,4,6,8,10,12,14]
        muscle_output = amps_all[:, body_left_idx] * (1 + np.cos(phases_all[:, body_left_idx]))

        # This is your "joint angle proxy" — shape (2000, 8)
        all_joint_pos = muscle_output
        times = np.arange(all_joint_pos.shape[0]) * timestep


        # Use only the second half to avoid transient startup
        half = len(times) // 2
        steady_pos = all_joint_pos[half:, :8]   # body joints only
        steady_times = times[half:]

        # ── Phase lag computation ────────────────────────────────────────────
        lags = compute_phase_lags_from_phases(
            state_array[:, :32],   # full phases array
            body_left_idx=np.arange(0, 16, 2)
        )
        # ── Dominant frequency ───────────────────────────────────────────────
        fft = np.abs(np.fft.rfft(steady_pos[:, 0]))
        freqs_fft = np.fft.rfftfreq(len(steady_pos), d=timestep)
        dom_freq = freqs_fft[np.argmax(fft[1:]) + 1]

        # And update dominant frequency to use CPG phases directly:
        phase_signal = state_array[len(times)//2:, 0]  # joint 0, second half
        # Frequency = d(phase)/dt / (2*pi)
        phase_rate = np.diff(phase_signal) / timestep
        dom_freq = np.mean(phase_rate) / (2 * np.pi)

        # ── Summary log ─────────────────────────────────────────────────────
        total_lag = np.sum(lags)
        pylog.info(
            f'\n{"="*55}\n'
            f'  Mode          : {label.upper()}  (drive={drive_val})\n'
            f'  Frequency     : {dom_freq:.3f} Hz\n'
            f'  Per-joint lags: {np.round(lags, 3)} rad\n'
            f'  Total lag     : {total_lag:.3f} rad  '
            f'({np.degrees(total_lag):.1f}°)\n'
            f'  Expected swim : {2*np.pi/8:.3f} rad/joint  '
            f'→ {np.degrees(2*np.pi/8):.1f}° per joint\n'
            f'{"="*55}'
        )

        results[label] = {
            'times': steady_times,
            'joint_positions': steady_pos,
            'lags': lags,
            'dominant_freq': dom_freq,
            'drive': drive_val,
        }

        plot_spine_analysis(steady_times, steady_pos, lags, drive_val, label)

    # ── Comparison plot: phase lags walking vs swimming ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(7)   # 7 pairs for 8 joints
    width = 0.35
    ax.bar(x - width/2, results['walking']['lags'],  width, label='Walking',  color='steelblue')
    ax.bar(x + width/2, results['swimming']['lags'], width, label='Swimming', color='coral')
    ax.axhline(2 * np.pi / 8, color='gray', linestyle='--', lw=1,
               label='Ideal swim lag (2π/8 ≈ 0.785 rad)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}→{k+1}' for k in range(7)])
    ax.set_ylabel('Phase lag [rad]')
    ax.set_title('Inter-joint phase lags: Walking vs Swimming')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./logs/ex3_1/comparison_walk_vs_swim.png', dpi=150)
    pylog.info('Saved comparison figure: logs/ex3_1/comparison_walk_vs_swim.png')
    plt.show()

    return results

def exercise_3_disable_limb_spine_coupling(timestep):
    """ Walk with disabled limb-spine limbs """
    # Use exercise_example.py for reference
    pass
    return


def exercise_3_limb_spine_antiphase(timestep):
    """ Walk with limb-spine in anti-phase """
    # Use exercise_example.py for reference
    pass
    return


def exercise_3a_coordination(timestep):
    """Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and limb oscillators.

    """
    # Use exercise_example.py for reference
    pass
    # # For sweeps with many simulations running in parallel
    # parameter_set = [
    #     SimulationParameters(...)
    #     for ... in ...
    #     for ... in ...
    # ]
    # os.makedirs('./logs/sweep_3a/', exist_ok=True)
    # simulation_sweep([
    #     {
    #         'sim_parameters': sim_parameters,
    #         'arena': 'land',
    #         'fast': True,  # For fast mode (not real-time)
    #         'headless': True,  # For headless mode (No GUI, could be faster)
    #         'output': f'logs/ex3a/simulation_{simulation_i}',
    #         'verbose': False,
    #     }
    #     for simulation_i, sim_parameters in enumerate(parameter_set)
    # ], processes=4)  # Adjust based on your hardware
    return


def exercise_3b_coordination(timestep):
    """Exercise 3b Limb and Spine coordination

    This exercise explores how axial and limb amplitudes affect coordination.

    Run the simulations for different axial and limb amplitudes.

    """
    # Use exercise_example.py for reference
    pass
    return


if __name__ == '__main__':
    exercise_3_1_spine_analysis(timestep=5e-3)
    exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    exercise_3_limb_spine_antiphase(timestep=5e-3)
    exercise_3a_coordination(timestep=5e-3)
    exercise_3b_coordination(timestep=5e-3)

