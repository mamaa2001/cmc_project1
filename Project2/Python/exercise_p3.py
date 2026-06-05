"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
from plot_results import *
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
from farms_amphibious.data.data import AmphibiousExperimentData





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
        (2.0,  'land',  'walking', 10),
        (4.0,  'water', 'swimming', 10),
        (None, 'land',  'ramp',    20),
    ]

    for drive_val, arena, label, duration in configs:
        print(f'Running {label} simulation (drive={drive_val}, arena={arena})')
        n_iter = int(duration / timestep)
        drive_input = np.linspace(0, 6, n_iter) if label == 'ramp' else drive_val

        sim_parameters = SimulationParameters(
            duration=duration,
            timestep=timestep,
            drive=drive_input,
            amplitude_gradient=None,
            phase_lag_body=None,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
        )

        os.makedirs(f'./logs/ex3_1/{label}/', exist_ok=True)
        _, sim_data = simulation(
            sim_parameters=sim_parameters,
            arena=arena,
            fast=True,
            headless=True,
            output=f'./logs/ex3_1/{label}/',
        )

        state_array = np.array(sim_data.state.array[:])
        times = np.arange(state_array.shape[0]) * timestep
        drive_for_plot = drive_input if label == 'ramp' else np.full(len(times), drive_val)

        plot_spine_analysis_paper_style(
            times=times,
            state_array=state_array,
            drive_val=drive_for_plot,
            mode_label=label,
            timestep=timestep,
        )
        if label in ('walking', 'swimming'):
            plot_phase_lags_analysis(
                state_array=state_array,
                drive_val=drive_val,
                mode_label=label,
                timestep=timestep,
            )
            plot_emg_style(
                state_array=state_array,
                drive_val=drive_val,
                mode_label=label,
                timestep=timestep,
            )


def exercise_3_disable_limb_spine_coupling(timestep):
    """ Walk with disabled limb-spine coupling (limb_spine_weight=0) """
    os.makedirs('./logs/ex3_no_coupling/', exist_ok=True)
    sim_parameters = SimulationParameters(
        duration=15,
        timestep=timestep,
        spawn_position=[0, 0, 0.1],
        spawn_orientation=[0, 0, np.pi/2],
        drive=2.5,
        limb_spine_weight=0,  # disable spine-limb coupling
    )
    simulation(
        sim_parameters=sim_parameters,
        arena='land',
        output='logs/ex3_no_coupling/sim_0',
        record=False,
        record_path='logs/ex3_no_coupling/video_no_coupling.mp4',
        verbose=True,
    )
    #return
    """
    Exercise 3.2 — Disable limb-spine coupling and compare with normal walking.

    Runs two walking simulations at drive=2.0:
      (a) Normal:    limb_spine_weight = 30  (default)
      (b) Decoupled: limb_spine_weight = 0

    For each case produces:
      • Paper-style CPG dynamics plot (panels A/B/C/D)
      • EMG-style burst activation diagram
      • Phase-lag analysis figure
    Then produces a combined 2×2 comparison figure.
    """
    drive    = 2.0
    duration = 15

    cases = [
        ('coupled',    30.0, 'Walking — with limb-spine coupling'),
        ('decoupled',   0.0, 'Walking — no limb-spine coupling'),
    ]

    results = {}

    for key, lsw, label in cases:
        print(f'Running: {label}')

        sim_parameters = SimulationParameters(
            duration=duration,
            timestep=timestep,
            drive=drive,
            amplitude_gradient=None,
            phase_lag_body=None,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            limb_spine_weight=lsw,
        )

        os.makedirs(f'./logs/ex3_2/{key}/', exist_ok=True)
        _, sim_data = simulation(
            sim_parameters=sim_parameters,
            arena='land',
            fast=True,
            headless=True,
            output=f'./logs/ex3_2/{key}/',
        )

        state_array = np.array(sim_data.state.array[:])
        n_iter = state_array.shape[0]
        times = np.arange(n_iter) * timestep

        # Forward speed: load head-link world positions from the saved HDF5 file
        
        exp_data  = AmphibiousExperimentData.from_file(
            f'./logs/ex3_2/{key}/simulation.hdf5'
        )
        links_pos = np.array(exp_data.animats[0].sensors.links.urdf_positions())
        elapsed = (links_pos.shape[0] - links_pos.shape[0] // 2) * timestep
        fwd_speed = float(
            np.linalg.norm(
                links_pos[-1, 0, :2] - links_pos[links_pos.shape[0] // 2, 0, :2]
            ) / elapsed
        )

        results[key] = dict(
            state_array=state_array,
            times=times,
            label=label,
            speed=fwd_speed,
            hdf5_path=f'./logs/ex3_2/{key}/simulation.hdf5',
        )

        print(f'  [{label}] forward speed = {fwd_speed:.4f} m/s')

        # ── Per-case plots ────────────────────────────────────────────────
        drive_vec = np.full(n_iter, drive)
        mode_label_map = {
            'coupled':   'walking_with_coupling',
            'decoupled': 'limbs_decoupled_from_body_walking',
        }
        mlabel = mode_label_map[key]

        plot_spine_analysis_paper_style(
            times=times,
            state_array=state_array,
            drive_val=drive_vec,
            mode_label=mlabel,
            timestep=timestep,
        )
        plot_emg_style(
            state_array=state_array,
            drive_val=drive,
            mode_label=mlabel,
            timestep=timestep,
        )
        plot_phase_lags_analysis(
            state_array=state_array,
            drive_val=drive,
            mode_label=mlabel,
            timestep=timestep,
        )

    plot_stability_comparison(results, timestep)
    return results


def exercise_3_limb_spine_antiphase(timestep, ideal_offset=0.0):
    """ Two videos: ideal phase offset vs. anti-phase (ideal + pi) """

    # ideal phase offset 
    os.makedirs('./logs/ex3_ideal/', exist_ok=True)
    simulation(
        sim_parameters=SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.5,
            limb_body_phase_offset=ideal_offset,
        ),
        arena='land',
        output='logs/ex3_ideal/sim_0',
        record=False,
        record_path='logs/ex3_ideal/video_ideal.mp4',
        verbose=True,
    )

    # anti-phase
    antiphase_offset = ideal_offset + np.pi
    os.makedirs('./logs/ex3_antiphase/', exist_ok=True)
    simulation(
        sim_parameters=SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.5,
            limb_body_phase_offset=antiphase_offset,
        ),
        arena='land',
        output='logs/ex3_antiphase/sim_0',
        record=False,
        record_path='logs/ex3_antiphase/video_antiphase.mp4',
        verbose=True,
    )
    return


def exercise_3a_coordination(timestep):
    """Exercise 3a Limb and Spine coordination

    This exercise explores how phase difference between spine and legs
    affects locomotion.

    Run the simulations for different walking drives and phase lag between body
    and limb oscillators.

    """
    # Use exercise_example.py for reference
    log_folder_pattern = './logs/sweep_3a/simulation_{}'

    # For sweeps with many simulations running in parallel
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            # Orientation in Euler angles [rad]
            spawn_orientation=[0, 0, np.pi/2],
            drive=drive,  # An example of parameter part of the grid search
            limb_body_phase_offset = limb_body_phase_offset,
        )
        for drive in np.linspace(1,5,25)
        for limb_body_phase_offset in np.linspace(-np.pi,np.pi,15)
    ]
    os.makedirs('./logs/sweep_3a/', exist_ok=True)
    simulation_sweep([
        {
            'sim_parameters': sim_parameters,
            'arena': 'land',
            'fast': True,  # For fast mode (not real-time)
            'headless': True,  # For headless mode (No GUI, could be faster)
        'output': log_folder_pattern.format(simulation_i),
        'verbose': False,
        }
        for simulation_i, sim_parameters in enumerate(parameter_set)
    ], processes=8)  # Adjust based on your hardware
    # 3. Chargement de toutes les données avec ta fonction load_data
    all_results = []
    
    for i in range(len(parameter_set)):
        # log_folder_pattern vaut './logs/sweep_3a/simulation_{}'
        # La fonction load_data va appliquer le .format(i) à l'intérieur
        exp_data, parameter = load_data(log_folder_pattern, i)
        
        # On stocke le résultat dans une liste pour pouvoir l'analyser ensuite
        all_results.append({
            'simulation_i': i,
            'data': exp_data,
            'parameters': parameter
        })
    return 


def analyze_exercise_3a_results(base_log_folder='./logs/sweep_3a/'):
    log_folder_pattern = os.path.join(base_log_folder, 'simulation_{}')
    
    sim_folders = [f for f in os.listdir(base_log_folder) if f.startswith('simulation_')]
    total_simulations = len(sim_folders)
    
    if total_simulations == 0:
        print("Aucune simulation trouvée.")
        return
        
    print(f"Analyse de {total_simulations} simulations en cours...")

    speed_results = []
    cot_results = []

    for i in range(total_simulations):
        try:
            # 1. Chargement des données
            exp_data, parameters = load_data(log_folder_pattern, i)
            # On nomme la variable animat_data au lieu de data pour ne pas l'écraser plus tard !
            animat_data = exp_data.animats[0] 

            d = parameters.drive
            offset = parameters.limb_body_phase_offset
            timestep = exp_data.timestep

            # 2. Conversion sécurisée en numpy arrays (comme dans ton exemple)
            links_positions = np.array(animat_data.sensors.links.urdf_positions())
            joints_torques = np.array(animat_data.sensors.joints.motor_torques_all())
            joints_velocities = np.array(animat_data.sensors.joints.velocities_all())

            # 3. SOLUTION ROBUSTE : On calcule la vitesse des liens à partir de leur position
            # Ça évite de chercher une fonction "velocities" qui n'existe peut-être pas dans ton API
            links_velocities = np.gradient(links_positions, axis=0) / timestep

            # 4. Calcul des métriques avec tes fonctions
            fwd_speed, lat_speed = compute_speed(links_positions, links_velocities)

            # 5. Calcul du Cost of Transport (CoT)
            cot = compute_mechanical_cot(
                    joints_torques=joints_torques,
                    joints_velocities=joints_velocities,
                    links_positions=links_positions,
                    timestep=timestep
                )

            speed_results.append([d, offset, fwd_speed])
            cot_results.append([d, offset, cot])

        except Exception as e:
            # Utile pour ne pas crasher si une des 441 simulations a échoué
            print(f"Erreur sur la simulation {i} : {e}")

    # Conversion finale en tableaux NumPy pour le plot
    speed_results = np.array(speed_results)
    cot_results = np.array(cot_results)

    # --- Affichage de la Heatmap de Vitesse ---
    plt.figure('Exercise 3a: Forward Speed', figsize=(8, 6))
    plot_2d(
        results=speed_results,
        labels=['Drive', 'Phase Offset (Limb-Body) [rad]', 'Forward Speed [m/s]'],
        cmap='viridis'
    )
    plt.title('Forward Speed with respect to Drive and Phase Offset')

    # --- Affichage de la Heatmap du Cost of Transport ---
    plt.figure('Exercise 3a: Cost of Transport', figsize=(8, 6))
    plot_2d(
        results=cot_results,
        labels=['Drive', 'Phase Offset (Limb-Body) [rad]', 'Cost of Transport'],
        cmap='inferno',
        log=True
    )
    plt.title('Cost of Transport with respect to Drive and Phase Offset')

    plt.show()


def exercise_3b_coordination(timestep):
    """Sweep over body and limb position gains."""
    log_folder_pattern = './logs/sweep_3b/simulation_{}'

    parameter_set = [
        SimulationParameters(
            duration=20,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.0,
            limb_body_phase_offset=0,
            position_body_gain=position_body_gain,
            position_limb_gain=position_limb_gain,
        )
        for position_body_gain in np.linspace(0, 3, 21)
        for position_limb_gain in np.linspace(0, 3, 21)
    ]
    os.makedirs('./logs/sweep_3b/', exist_ok=True)
    simulation_sweep([
        {
            'sim_parameters': sim_parameters,
            'arena': 'land',
            'fast': True,
            'headless': True,
            'output': log_folder_pattern.format(simulation_i),
            'verbose': False,
        }
        for simulation_i, sim_parameters in enumerate(parameter_set)
    ], processes=8)

    all_results = []
    for i in range(len(parameter_set)):
        exp_data, parameter = load_data(log_folder_pattern, i)
        all_results.append({
            'simulation_i': i,
            'data': exp_data,
            'parameters': parameter
        })
    return


def analyze_exercise_3b_results(base_log_folder='./logs/sweep_3b/'):
    log_folder_pattern = os.path.join(base_log_folder, 'simulation_{}')
    sim_folders = [f for f in os.listdir(base_log_folder) if f.startswith('simulation_')]
    total_simulations = len(sim_folders)

    if total_simulations == 0:
        print("No simulations found.")
        return

    print(f"Analyzing {total_simulations} simulations...")

    speed_results = []
    cot_results = []

    for i in range(total_simulations):
        try:
            exp_data, parameters = load_data(log_folder_pattern, i)
            animat_data = exp_data.animats[0]
            timestep = exp_data.timestep

            body_gain = parameters.position_body_gain
            limb_gain = parameters.position_limb_gain

            links_positions = np.array(animat_data.sensors.links.urdf_positions())
            joints_torques = np.array(animat_data.sensors.joints.motor_torques_all())
            joints_velocities = np.array(animat_data.sensors.joints.velocities_all())

            com_initial = np.mean(links_positions[0, :9, :2], axis=0)
            com_final = np.mean(links_positions[-1, :9, :2], axis=0)
            distance_fwd = np.linalg.norm(com_final - com_initial)
            total_time = links_positions.shape[0] * timestep
            walking_speed = distance_fwd / total_time

            cot = compute_mechanical_cot(
                joints_torques=joints_torques,
                joints_velocities=joints_velocities,
                links_positions=links_positions,
                timestep=timestep
            )

            speed_results.append([body_gain, limb_gain, walking_speed])
            cot_results.append([body_gain, limb_gain, cot])

        except Exception as e:
            print(f"Error on simulation {i} : {e}")

    speed_results = np.array(speed_results)
    cot_results = np.array(cot_results)

    plt.figure('Gains Sweep: Walking Speed', figsize=(8, 6))
    plot_2d(
        results=speed_results,
        labels=['Body Gain', 'Limb Gain', 'Walking Speed [m/s]'],
        cmap='viridis'
    )
    plt.title('Walking Speed with respect to Body and Limb Gains')

    plt.figure('Gains Sweep: Cost of Transport', figsize=(8, 6))
    plot_2d(
        results=cot_results,
        labels=['Body Gain', 'Limb Gain', 'Cost of Transport'],
        cmap='inferno',
        log=True
    )
    plt.title('Cost of Transport with respect to Body and Limb Gains')

    plt.show()


def exercise_3b_optimal_video(timestep, optimal_body_gain=1.0, optimal_limb_gain=1.0, label='optimal'):
    """Video of the optimal oscillator amplitudes found in the 3b sweep."""
    folder = f'./logs/ex3b_{label}/'
    os.makedirs(folder, exist_ok=True)
    simulation(
        sim_parameters=SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2,
            limb_body_phase_offset=0.0,
            position_body_gain=optimal_body_gain,
            position_limb_gain=optimal_limb_gain,
        ),
        arena='land',
        output=f'{folder}sim_0',
        record=False,
        record_path=f'{folder}video_{label}.mp4',
        verbose=True,
    )
    return


if __name__ == '__main__':

    # For videos : 
    
    exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    exercise_3_limb_spine_antiphase(timestep=5e-3)
    exercise_3b_optimal_video(timestep=5e-3, optimal_body_gain=2.5, optimal_limb_gain=2.2, label='speed_optimal')
    exercise_3b_optimal_video(timestep=5e-3, optimal_body_gain=1.0, optimal_limb_gain=1.0, label='cot_optimal')
    #grid search
    #exercise_3a_coordination(timestep=5e-3)
    analyze_exercise_3a_results()
    #exercise_3b_coordination(timestep=5e-3)
    analyze_exercise_3b_results()