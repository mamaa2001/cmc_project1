"""Plot results"""

import os
import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from simulation_parameters import SimulationParameters
from farms_amphibious.data.data import AmphibiousExperimentData

from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from network import motor_output
import matplotlib.colors as colors
from matplotlib.patches import Patch


def load_data(
        log_files: str,
        simulation_i: int,
) -> tuple[AmphibiousExperimentData, SimulationParameters]:
    """Load data"""
    experiment_data_file = os.path.join(
        log_files.format(simulation_i),
        'simulation.hdf5',
    )
    exp_data = AmphibiousExperimentData.from_file(experiment_data_file)
    sim_parameters_file = os.path.join(
        log_files.format(simulation_i),
        'sim_parameters.pickle',
    )
    with open(sim_parameters_file, 'rb') as param_file:
        parameters = pickle.load(param_file)
    return exp_data, parameters


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)


def plot_trajectory(link_data, label=None, color=None):
    """Plot trajectory"""
    plt.plot(link_data[:, 0], link_data[:, 1], label=label, color=color)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def max_distance(link_data, nsteps_considered=None):
    """Compute max distance"""
    if not nsteps_considered:
        nsteps_considered = link_data.shape[0]
    com = np.mean(link_data[-nsteps_considered:], axis=1)
    return np.sqrt(
        np.max(np.sum((link_data[:, :]-link_data[0, :])**2, axis=1)))


def compute_speed(links_positions, links_vel, nsteps_considered=200):
    """
    Computes the axial and lateral speed based on the PCA of the links positions
    """

    links_pos_xy = links_positions[-nsteps_considered:, :, :2]
    joints_vel_xy = links_vel[-nsteps_considered:, :, :2]
    time_idx = links_pos_xy.shape[0]

    speed_forward = []
    speed_lateral = []
    com_pos = []

    for idx in range(time_idx):
        x = links_pos_xy[idx, :9, 0]
        y = links_pos_xy[idx, :9, 1]

        pheadtail = links_pos_xy[idx][0]-links_pos_xy[idx][8]  # head - tail
        pcom_xy = np.mean(links_pos_xy[idx, :9, :], axis=0)
        vcom_xy = np.mean(joints_vel_xy[idx], axis=0)

        covmat = np.cov([x, y])
        eig_values, eig_vecs = np.linalg.eig(covmat)
        largest_index = np.argmax(eig_values)
        largest_eig_vec = eig_vecs[:, largest_index]

        ht_direction = np.sign(np.dot(pheadtail, largest_eig_vec))
        largest_eig_vec = ht_direction * largest_eig_vec

        v_com_forward_proj = np.dot(vcom_xy, largest_eig_vec)

        left_pointing_vec = np.cross(
            [0, 0, 1],
            [largest_eig_vec[0], largest_eig_vec[1], 0]
        )[:2]

        v_com_lateral_proj = np.dot(vcom_xy, left_pointing_vec)

        com_pos.append(pcom_xy)
        speed_forward.append(v_com_forward_proj)
        speed_lateral.append(v_com_lateral_proj)

    return np.mean(speed_forward), np.mean(speed_lateral)


def sum_torques(joints_data):
    """Compute sum of torques

    Example:

    joints_data = data.sensors.joints.motor_torques_all()

    """
    return np.sum(np.abs(joints_data[:, :]))


def compute_mechanical_cot(joints_torques, joints_velocities, links_positions, timestep):
    """
    Calcule le Cost of Transport basé sur le vrai travail mécanique positif.
    """
    power = np.maximum(joints_torques * joints_velocities, 0)
    total_energy = timestep * np.sum(power)
    com_initial = np.mean(links_positions[0, :9, :2], axis=0)
    com_final = np.mean(links_positions[-1, :9, :2], axis=0)
    distance_fwd = np.linalg.norm(com_final - com_initial)
    if distance_fwd > 0.001:
        cot = total_energy / distance_fwd
    else:
        cot = np.nan
    return cot


def plot_network_dynamics(times, phases_log, amplitudes_log, freqs_log, drive_vec, title='Network dynamics (drive ramp 0→6)'):
    """
    Plots the network dynamics mirroring Figures 4 & 5 from Ijspeert 2007.
    """

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title)

    osc_left = np.arange(0, 16, 2)

    # Panel A: body oscillator outputs  x_i = r_i(1 + cos(φ_i))
    body_output = amplitudes_log[:, :16] * (1 + np.cos(phases_log[:, :16]))
    for idx in osc_left:
        axes[0].plot(times, body_output[:, idx], lw=0.8)
    axes[0].set_ylabel('x Body (left)')
    axes[0].set_title('A – Body oscillator outputs')

    # Panel B: limb oscillator outputs
    limb_output = amplitudes_log[:, 16:] * (1 + np.cos(phases_log[:, 16:]))
    for idx in range(0, 16, 4):
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
    
    return fig, axes


def plot_spine_analysis_paper_style(times, state_array, drive_val, mode_label, timestep):
    """
    Reproduce Figure 2 style from Ijspeert 2007:
    A - x_i signals from left body oscillators (muscle outputs)
    B - x_i signals from left limb oscillators  
    C - Instantaneous frequencies
    D - Drive signal
    """
    if mode_label == 'ramp':
        drive_label = 'ramp 0→6'
    else:
        drive_label = f'{float(np.asarray(drive_val).flat[0]):.1f}'
    title_label = mode_label.replace('_', ' ')
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'CPG dynamics — {title_label}  (drive={drive_label})', fontsize=13)

    phases = state_array[:, :32]
    amps   = state_array[:, 32:64]

    x = amps * (1 + np.cos(phases))

    dphi = np.diff(phases, axis=0) / timestep
    dphi = np.where(dphi > 0, dphi, np.nan)
    inst_freq = dphi / (2 * np.pi)
    times_freq = times[1:]

    body_left  = np.arange(0, 16, 2)

    # ── Panel A: body oscillator outputs (left side) ─────────────────────────
    ax = axes[0]
    colors_body = plt.cm.viridis(np.linspace(0, 1, len(body_left)))
    for idx, osc_i in enumerate(body_left):
        # Offset each trace vertically like in the paper (stacked)
        offset = (len(body_left) - 1 - idx) * 0.8
        ax.plot(times, x[:, osc_i] + offset,
                color=colors_body[idx], lw=0.8)
        ax.text(-0.5, offset + 0.3, f'x{osc_i}', fontsize=7,
                ha='right', va='center', color=colors_body[idx])
    ax.set_ylabel('x Body (left)')
    ax.set_title('A')
    ax.set_yticks([])


    # ── Panel B: limb oscillator outputs ─────────────────────────────────────
    ax = axes[1]
    # Show one oscillator per limb (the girdle+ of each)
    limb_names = ['FL-L', 'FL-R', 'HL-L', 'HL-R']
    limb_show  = [16, 20, 24, 28]
    colors_limb = ['steelblue', 'tomato', 'green', 'orange']
    for idx, (osc_i, name) in enumerate(zip(limb_show, limb_names)):
        offset = (len(limb_show) - 1 - idx) * 0.8
        ax.plot(times, x[:, osc_i] + offset,
                color=colors_limb[idx], lw=0.8, label=name)
        ax.text(-0.5, offset + 0.3, name, fontsize=7,
                ha='right', va='center', color=colors_limb[idx])
    ax.set_ylabel('x Limb')
    ax.set_title('B')
    ax.set_yticks([])


    # ── Panel C: instantaneous frequencies ───────────────────────────────────
    ax = axes[2]
    # Body: mean over left body oscillators
    freq_body = np.nanmean(inst_freq[:, body_left], axis=1)
    # Limb: mean over limb girdle oscillators
    freq_limb = np.nanmean(inst_freq[:, [16, 20, 24, 28]], axis=1)
    # Smooth with a short window to reduce noise
    def smooth(x, w=50):
        return np.convolve(x, np.ones(w)/w, mode='same')
    ax.plot(times_freq, smooth(freq_body), 'k-',  lw=1.2, label='Body')
    ax.plot(times_freq, smooth(freq_limb), 'k--', lw=1.2, label='Limb')
    ax.set_ylabel('Freq [Hz]')
    ax.set_title('C')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    # ── Panel D: drive ────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(times, drive_val, 'k-', lw=1.2)
    ax.axhline(1.0, color='red', linestyle='--', lw=0.8, label='d_low=1')
    ax.axhline(3.0, color='red', linestyle='--', lw=0.8, label='d_high limb=3')
    ax.axhline(5.0, color='red', linestyle='--', lw=0.8, label='d_high body=5')
    ax.set_ylabel('drive d')
    ax.set_xlabel('Time [s]')
    ax.set_title('D')
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs('./logs/ex3_1/', exist_ok=True)
    fname = f'./logs/ex3_1/fig2_style_{mode_label}.png'
    plt.savefig(fname, dpi=150)
    print(f'Saved: {fname}')
    plt.show()

def plot_stability_comparison(results, timestep):
    """
    Compare lateral deviation and limb-oscillator regularity between coupled and
    decoupled walking.

    All metrics come from the saved HDF5 file (smoothed contact forces).

    Figure 1 (3 rows × 2 columns):
      Row 0 — head trajectory (top-down)
      Row 1 — lateral deviation over time
      Row 2 — gait diagram (smoothed foot contacts)

    Figure 2 — summary bar charts: lateral-deviation σ | foot contact CV | speed.
    """
    keys   = ['coupled', 'decoupled']
    colors = {'coupled': 'steelblue', 'decoupled': 'tomato'}

    sensor_data = {}
    for key in keys:
        # ── Foot contact CV from HDF5 (smoothed to remove physics noise) ────────
        foot_idx   = [10, 12, 14, 16]
        threshold  = 0.5
        win        = max(1, int(0.05 / timestep))
        animat     = AmphibiousExperimentData.from_file(
            results[key]['hdf5_path']
        ).animats[0]
        links      = np.array(animat.sensors.links.urdf_positions())
        contacts   = np.array(animat.sensors.contacts.totals())
        n_links    = links.shape[0]
        half_l     = n_links // 2

        forces_all  = np.linalg.norm(contacts[half_l:], axis=2)
        feet_forces = forces_all[:, foot_idx]

        foot_active = []
        for fi in range(4):
            fc_smooth = np.convolve(feet_forces[:, fi], np.ones(win) / win, mode='same')
            foot_active.append(fc_smooth > threshold)

        # ── Body-limb phase coordination (circular std of body phase at foot strike)
        # At each foot contact onset, sample the phase of the corresponding body
        # oscillator. Circular std ≈ 0 → foot always strikes at the same body
        # phase (good coordination); circular std → π → random (no coordination).
        sa        = results[key]['state_array']
        half_sa   = sa.shape[0] // 2
        phases_ss = sa[half_sa:, :32]

        limb_body_pairs = [(0, 0), (1, 1), (2, 8), (3, 9)]
        coord_stds = []
        for foot_i, body_osc in limb_body_pairs:
            starts = np.where(np.diff(
                np.concatenate([[False], foot_active[foot_i], [False]]).astype(int)
            ) == 1)[0]
            starts = starts[starts < len(phases_ss)]
            if len(starts) > 1:
                body_ph = np.mod(phases_ss[starts, body_osc], 2 * np.pi)
                R       = np.abs(np.mean(np.exp(1j * body_ph)))
                coord_stds.append(float(np.sqrt(-2 * np.log(R + 1e-9))))
            else:
                coord_stds.append(np.nan)

        # ── Lateral deviation from head link positions ────────────────────────
        head_xy = links[half_l:, 0, :2]
        disp    = head_xy[-1] - head_xy[0]
        fwd     = disp / (np.linalg.norm(disp) + 1e-9)
        lat     = np.array([-fwd[1], fwd[0]])
        lat_dev = (head_xy - head_xy[0]) @ lat

        sensor_data[key] = dict(
            coord_stds  = np.array(coord_stds),
            head_xy     = head_xy,
            lat_dev     = lat_dev,
            lat_std     = np.std(lat_dev),
        )

    t_head = np.arange(sensor_data[keys[0]]['lat_dev'].shape[0]) * timestep

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle('Stability comparison: with vs without limb-spine coupling', fontsize=13)

    for col, key in enumerate(keys):
        sd    = sensor_data[key]
        label = results[key]['label']
        color = colors[key]

        # Row 0: head trajectory
        ax = axes[0, col]
        ax.plot(sd['head_xy'][:, 0], sd['head_xy'][:, 1], color=color, lw=1.0)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel('y [m]  (head trajectory)')
        ax.text(0.03, 0.97, f'Lateral σ = {sd["lat_std"]*100:.2f} cm',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))

        # Row 1: lateral deviation over time
        ax1 = axes[1, col]
        ax1.plot(t_head, sd['lat_dev'] * 100, color=color, lw=0.8)
        ax1.fill_between(t_head, sd['lat_dev'] * 100, 0, alpha=0.15, color=color)
        ax1.axhline(0, color='k', lw=0.5, linestyle='--')
        ax1.set_xlabel('Time [s]')
        ax1.set_title(f'Lateral deviation  (σ = {sd["lat_std"]*100:.2f} cm)')
        if col == 0:
            ax1.set_ylabel('Lateral deviation [cm]')

    plt.tight_layout()
    os.makedirs('./logs/ex3_2/', exist_ok=True)
    plt.savefig('./logs/ex3_2/stability_comparison.png', dpi=150)
    print('Saved: logs/ex3_2/stability_comparison.png')
    plt.show()

    # ── Figure 2: summary bar charts (lateral dev | foot contact CV | speed) ──
    bar_labels     = ['With coupling', 'No coupling']
    bar_color_list = [colors['coupled'], colors['decoupled']]

    def _bar_panel(ax, values, ylabel, title, fmt):
        bars = ax.bar(bar_labels, values, color=bar_color_list, edgecolor='k', width=0.4)
        vmax = max(v for v in values if not (isinstance(v, float) and np.isnan(v)))
        for bar, v in zip(bars, values):
            if not (isinstance(v, float) and np.isnan(v)):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + vmax * 0.04, fmt.format(v),
                        ha='center', fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, vmax * 1.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    _, (ax_lat, ax_cv, ax_spd) = plt.subplots(1, 3, figsize=(13, 4))
    plt.suptitle('Stability and speed metrics summary', fontsize=12)

    lat_stds = [sensor_data[k]['lat_std'] * 100 for k in keys]
    _bar_panel(ax_lat, lat_stds,
               'Lateral deviation σ [cm]', 'Path straightness  (lower = straighter)',
               '{:.2f} cm')

    mean_coord = [float(np.nanmean(sensor_data[k]['coord_stds'])) for k in keys]
    _bar_panel(ax_cv, mean_coord,
               'Body phase std at foot strike [rad]',
               'Limb-body coordination  (lower = better)',
               '{:.3f} rad')

    speeds = [results[k]['speed'] for k in keys]
    _bar_panel(ax_spd, speeds,
               'Forward speed [m/s]', 'Walking speed  (higher = faster)',
               '{:.4f} m/s')

    plt.tight_layout()
    plt.savefig('./logs/ex3_2/stability_metrics.png', dpi=150)
    print('Saved: logs/ex3_2/stability_metrics.png')
    plt.show()

def plot_phase_lags_analysis(state_array, drive_val, mode_label, timestep):
    """
    Dedicated figure to answer: 'What are the phase lags along the spine?'

    Panel A — Spatial phase profile: phase of each left body oscillator
               relative to J0 at a reference instant in steady state.
    Panel B — Inter-joint phase lags (bar chart, in degrees) with annotated
               values, compared to the ideal swimming lag (45°/joint).
    Text box — Mean lag, total lag, wavelength, and dominant frequency.
    """
    phases = state_array[:, :32]          # [n_iter, 32]

    body_left_idx = np.arange(0, 16, 2)  # [0,2,4,...,14]  — 8 joints

    # ── Steady-state window (second half) ────────────────────────────────────
    half = phases.shape[0] // 2
    phases_ss = phases[half:, :]

    # ── Per-joint phase lags ──────────────────────────────────────────────────
    phases_body = phases_ss[:, body_left_idx]   # [n_ss, 8]
    lags = []
    for k in range(7):
        diff = phases_body[:, k] - phases_body[:, k + 1]
        lags.append(np.mean(np.arctan2(np.sin(diff), np.cos(diff))))
    lags = np.array(lags)

    # ── Spatial phase profile (relative to J0) at mid-steady-state ───────────
    ref_idx   = phases_body.shape[0] // 2
    phases_ref = np.mod(phases_body[ref_idx, :], 2 * np.pi)
    phases_rel = np.degrees(np.mod(phases_ref - phases_ref[0], 2 * np.pi))
    # for plot continuity
    phases_rel[0] = 360.0 if phases_rel[1] > 180 else 0.0

    # ── Dominant frequency from CPG phase rate ────────────────────────────────
    dphase    = np.diff(phases_ss[:, 0]) / timestep   # rad/s
    dom_freq  = np.nanmean(dphase[dphase > 0]) / (2 * np.pi)

    # ── Summary stats ─────────────────────────────────────────────────────────
    mean_lag_deg  = np.degrees(np.mean(lags)) 
    total_lag_deg = np.degrees(np.sum(lags))
    pattern = 'TRAVELING WAVE' if mode_label == "swimming" else 'STANDING WAVE'

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    fig.suptitle(
        f'Phase-lag analysis along spine — {mode_label}  (drive={drive_val})',
        fontsize=13
    )

    # Panel A: spatial phase profile
    ax = axes[0]
    joints = np.arange(8)
    ax.plot(joints, phases_rel, 'o-', color='steelblue', lw=1.8, ms=8, zorder=3)
    fill_ref = 360 if phases_rel[0] == 360.0 else 0
    ax.fill_between(joints, phases_rel, fill_ref, alpha=0.15, color='steelblue')
    ax.axhline(180, color='gray', lw=0.9, linestyle=':', label='180° (anti-phase to J0)')
    ax.axhline(0,   color='black', lw=0.5, linestyle='--')
    ax.set_xticks(joints)
    ax.set_xticklabels([f'J{j}' for j in joints])
    ax.set_ylabel('Phase relative to J0 [°]')
    ax.set_xlabel('Body joint  (head → tail)')
    ax.set_title('A — Spatial phase profile along spine')
    ax.set_ylim(-10, 370)
    ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.legend(fontsize=8)
    # Annotate each point
    for j, p in enumerate(phases_rel):
        ax.annotate(f'{p:.0f}°', (j, p), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    
    # Panel B: inter-joint phase lags bar chart
    ax2 = axes[1]
    lags_deg   = np.degrees(lags)
    pair_labels = [f'J{k}→J{k+1}' for k in range(7)]
    bar_colors  = ['steelblue' if v >= 0 else 'tomato' for v in lags_deg]
    bars = ax2.bar(pair_labels, lags_deg, color=bar_colors,
                   edgecolor='k', linewidth=0.6, zorder=3)
    ax2.axhline(0, color='black', lw=0.8)
    if mode_label == "swimming" :
        ax2.axhline(45, color='coral', lw=1.2, linestyle='--',
                    label='Ideal swim lag (360°/8 = 45°/joint)')
        ax2.axhline(-45, color='coral', lw=1.2, linestyle='--')
    ax2.set_ylabel('Phase lag [°]')
    ax2.set_xlabel('Joint pair  (head → tail)')
    ax2.set_title('B — Inter-joint phase lags ')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    # Annotate bars
    for bar, val in zip(bars, lags_deg):
        ypos = val + (3 if val >= 0 else -10)
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f'{val:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Summary text box
    summary = (
        f'Total lag J0→J7  : {total_lag_deg:.1f}°\n'
        f'Frequency        : {dom_freq:.3f} Hz\n'
        f'Pattern          : {pattern}'
    )
    ax2.text(0.02, 0.97, summary, transform=ax2.transAxes,
             fontsize=8.5, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='gray', alpha=0.9))
 
    plt.tight_layout()
    os.makedirs('./logs/ex3_1/', exist_ok=True)
    fname = f'./logs/ex3_1/phase_lag_analysis_{mode_label}.png'
    plt.savefig(fname, dpi=150)
    print(f'Saved: {fname}')
    plt.show()

    # Print clear answer to the assignment question
    print(
        f'\n{"="*60}\n'
        f'  PHASE LAGS ALONG THE SPINE — {mode_label.upper()}\n'
        f'{"="*60}\n'
        f'  Drive            : {drive_val}\n'
        f'  Per-joint lags   : {np.round(lags_deg, 1)} °\n'
        f'  Mean lag/joint   : {mean_lag_deg:.1f}°  ({np.radians(mean_lag_deg):.3f} rad)\n'
        f'  Total lag J0→J7  : {total_lag_deg:.1f}°  ({np.radians(total_lag_deg):.3f} rad)\n'
        f'  Frequency        : {dom_freq:.3f} Hz\n'
        f'  Wave pattern     : {pattern}\n'
        f'{"="*60}'
    )


def _fill_bursts(ax, t, active, y_low, y_high, color, alpha=0.82):
    """Fill each continuous active epoch with a colored rectangle."""
    padded  = np.concatenate([[False], np.asarray(active, bool), [False]])
    changes = np.diff(padded.astype(int))
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        ax.fill_between(
            [t[min(s, len(t)-1)], t[min(e, len(t)-1)]],
            y_low, y_high,
            color=color, alpha=alpha, linewidth=0,
        )


def plot_emg_style(state_array, drive_val, mode_label, timestep, n_cycles=3):
    """
    EMG-style burst activation diagram (style of Cabelguen et al. recordings).

    Layout head→tail: Forelimb | Spine S1-S4 | Hindlimb | Spine S5-S8
    Each row has two bands: left muscle (green, upper) / right muscle (red, lower).
    Dashed vertical lines mark every half-period for easy phase reading.
    Silent limbs (swimming) are shown as a dim tonic bar.
    """
    phases = state_array[:, :32]
    amps   = state_array[:, 32:64]
    x      = amps * (1 + np.cos(phases))

    half     = x.shape[0] // 2
    x_ss     = x[half:, :]
    amps_ss  = amps[half:, :]
    phases_ss = phases[half:, :]
    t_ss     = np.arange(x_ss.shape[0]) * timestep

    # Period from body oscillator 0
    dphase = np.diff(phases_ss[:, 0]) / timestep
    period = 2 * np.pi / np.nanmean(dphase[dphase > 0])

    n_show = min(int(n_cycles * period / timestep), x_ss.shape[0])
    x_plt  = x_ss[:n_show, :]
    a_plt  = amps_ss[:n_show, :]
    t_plt  = t_ss[:n_show]

    # ── Row definitions (head at top): label, osc_L, osc_R, is_limb ─────────
    rows = [
        ('FL',  16, 20, True),
        ('S1',   0,  1, False),
        ('S2',   2,  3, False),
        ('S3',   4,  5, False),
        ('S4',   6,  7, False),
        ('HL',  24, 28, True),
        ('S5',   8,  9, False),
        ('S6',  10, 11, False),
        ('S7',  12, 13, False),
        ('S8',  14, 15, False),
    ]

    row_h    = 0.28   # height of one muscle band
    gap      = 0.04   # gap between upper (L) and lower (R) band
    row_sep  = 0.88   # pitch between row centres
    limb_sep = 0.22   # extra spacing around limb rows

    # Assign y-centres bottom-to-top so head ends up at the top of the plot
    y_centers = []
    y = 0.0
    for i in range(len(rows) - 1, -1, -1):
        y_centers.insert(0, y)
        extra = limb_sep if (rows[i][3] or (i > 0 and rows[i-1][3])) else 0
        y += row_sep + extra

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        f'EMG-style muscle activations — {mode_label.upper()}  (drive={drive_val})',
        fontsize=13,
    )

    for (label, osc_L, osc_R, is_limb), y_c in zip(rows, y_centers):
        y_L_lo, y_L_hi = y_c + gap/2,          y_c + gap/2 + row_h
        y_R_lo, y_R_hi = y_c - gap/2 - row_h,  y_c - gap/2

        limb_silent = is_limb and (a_plt[:, osc_L].mean() < 0.01)

        if limb_silent:
            # Tonic retraction during swimming — dim filled bar
            ax.fill_between(t_plt, y_L_lo, y_L_hi, color='green', alpha=0.25)
            ax.fill_between(t_plt, y_R_lo, y_R_hi, color='red',   alpha=0.25)
            ax.text(t_plt[-1]/2, y_c, 'tonic (retracted)',
                    ha='center', va='center', fontsize=7, color='gray', style='italic')
        else:
            _fill_bursts(ax, t_plt, x_plt[:, osc_L] > np.mean(x_plt[:, osc_L]),
                         y_L_lo, y_L_hi, 'green')
            _fill_bursts(ax, t_plt, x_plt[:, osc_R] > np.mean(x_plt[:, osc_R]),
                         y_R_lo, y_R_hi, 'red')

        ax.axhline(y_c, color='lightgray', lw=0.4, zorder=0)
        ax.text(-0.04 * t_plt[-1], y_c, label,
                fontsize=9, ha='right', va='center',
                fontweight='bold' if is_limb else 'normal',
                color='#1a6b1a' if is_limb else 'black')

    # ── Half-period dashed reference lines ───────────────────────────────────
    for t_ref in np.arange(0, t_plt[-1], period / 2):
        ax.axvline(t_ref, color='black', lw=0.6, linestyle='--', alpha=0.35, zorder=0)

    # ── Period marker T ──────────────────────────────────────────────────────
    y_bot = min(y_centers) - 0.65
    ax.annotate('', xy=(period, y_bot), xytext=(0, y_bot),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(period / 2, y_bot - 0.28, 'T', ha='center', fontsize=11, fontweight='bold')

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    ax.set_xlim(-0.05 * t_plt[-1], t_plt[-1])
    ax.set_ylim(y_bot - 0.45, max(y_centers) + 0.55)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_yticks([])
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(
        handles=[Patch(color='green', label='Left muscle'),
                 Patch(color='red',   label='Right muscle')],
        fontsize=9, loc='upper right',
    )

    plt.tight_layout()
    os.makedirs('./logs/ex3_1/', exist_ok=True)
    fname = f'./logs/ex3_1/emg_style_{mode_label}.png'
    plt.savefig(fname, dpi=150)
    print(f'Saved: {fname}')
    plt.show()

def main(plot=True):
    """Main"""
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=save_plots())

