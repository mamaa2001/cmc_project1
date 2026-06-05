"""Exercise 3: Limb and Spine Coordination while walking"""

import os
import numpy as np
from plot_results import *
from salamandra_simulation.simulation import simulation, simulation_sweep
from simulation_parameters import SimulationParameters
#import farms_pylog as pylog
import matplotlib.pyplot as plt
from farms_amphibious.data.data import AmphibiousExperimentData
from matplotlib.patches import Patch


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
    phases_ss = phases[half:, :]          # [n_ss, 32]

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
    fill_ref = 360 if phases_rel[0] == 360.0 else 0   # for plot continuity
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
    x      = amps * (1 + np.cos(phases))   # [n_iter, 32]

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
        ('FL',  16, 20, True),    # Forelimb shoulder oscillators
        ('S1',   0,  1, False),
        ('S2',   2,  3, False),
        ('S3',   4,  5, False),
        ('S4',   6,  7, False),
        ('HL',  24, 28, True),    # Hindlimb shoulder oscillators
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

        state_array    = np.array(sim_data.state.array[:])
        times          = np.arange(state_array.shape[0]) * timestep
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

    phases = state_array[:, :32]      # shape [n_iter, 32]
    amps   = state_array[:, 32:64]    # shape [n_iter, 32]

    # Muscle output: x_i = r_i * (1 + cos(phi_i))
    x = amps * (1 + np.cos(phases))   # shape [n_iter, 32]

    # Instantaneous frequency: f_i = d(phi_i)/dt / (2*pi)
    dphi = np.diff(phases, axis=0) / timestep          # shape [n_iter-1, 32]
    # Wrap to avoid 2pi jumps
    dphi = np.where(dphi > 0, dphi, np.nan)            # phase only increases
    inst_freq = dphi / (2 * np.pi)                     # in Hz
    times_freq = times[1:]

    body_left  = np.arange(0, 16, 2)   # [0,2,4,6,8,10,12,14]

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
    ax.plot(times, drive_val, 'k-', lw=1.2)   # already an array from the caller
    # Threshold lines like in paper
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
        win        = max(1, int(0.05 / timestep))   # 50 ms smoothing window
        animat     = AmphibiousExperimentData.from_file(
            results[key]['hdf5_path']
        ).animats[0]
        links      = np.array(animat.sensors.links.urdf_positions())
        contacts   = np.array(animat.sensors.contacts.totals())
        n_links    = links.shape[0]
        half_l     = n_links // 2

        forces_all  = np.linalg.norm(contacts[half_l:], axis=2)   # [n_ss, n_links]
        feet_forces = forces_all[:, foot_idx]                       # [n_ss, 4]

        foot_active = []   # smoothed binary contact per foot, for gait diagram
        for fi in range(4):
            fc_smooth = np.convolve(feet_forces[:, fi], np.ones(win) / win, mode='same')
            foot_active.append(fc_smooth > threshold)

        # ── Body-limb phase coordination (circular std of body phase at foot strike)
        # At each foot contact onset, sample the phase of the corresponding body
        # oscillator. Circular std ≈ 0 → foot always strikes at the same body
        # phase (good coordination); circular std → π → random (no coordination).
        sa        = results[key]['state_array']
        half_sa   = sa.shape[0] // 2
        phases_ss = sa[half_sa:, :32]   # steady-state oscillator phases

        # foot index → nearest body oscillator (left/right pair at girdle)
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
        record=True,
        record_path='logs/ex3_no_coupling/video_no_coupling.mp4',
        verbose=True,
    )
    return
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

    cases = [ #key, lsw, label
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
        n_iter      = state_array.shape[0]
        times       = np.arange(n_iter) * timestep

        # Forward speed: load head-link world positions from the saved HDF5 file
        
        exp_data  = AmphibiousExperimentData.from_file(
            f'./logs/ex3_2/{key}/simulation.hdf5'
        )
        links_pos = np.array(exp_data.animats[0].sensors.links.urdf_positions())
        elapsed   = (links_pos.shape[0] - links_pos.shape[0] // 2) * timestep
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
        record=True,
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
        record=True,
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
    """Exercise 3b Limb and Spine coordination

    This exercise explores how axial and limb amplitudes affect coordination.

    Run the simulations for different axial and limb amplitudes.

    """
    # Use exercise_example.py for reference
    # Use exercise_example.py for reference
    log_folder_pattern = './logs/sweep_3b/simulation_{}'

    # For sweeps with many simulations running in parallel
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            # Orientation in Euler angles [rad]
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.0,  # from previous grid search
            limb_body_phase_offset = 0, #from previous grid search
            position_body_gain = position_body_gain,
            position_limb_gain = position_limb_gain,
        )
        for position_body_gain in np.linspace(0,3,21)
        for position_limb_gain in np.linspace(0,3,21)
    ]
    os.makedirs('./logs/sweep_3b/', exist_ok=True)
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

def analyze_exercise_3b_results(base_log_folder='./logs/sweep_3b/'):
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
            exp_data, parameters = load_data(log_folder_pattern, i)
            animat_data = exp_data.animats[0] 
            timestep = exp_data.timestep

            # --- NOUVEAUX PARAMÈTRES EXTRAITS ICI ---
            body_gain = parameters.position_body_gain
            limb_gain = parameters.position_limb_gain

            links_positions = np.array(animat_data.sensors.links.urdf_positions())
            joints_torques = np.array(animat_data.sensors.joints.motor_torques_all())
            joints_velocities = np.array(animat_data.sensors.joints.velocities_all())

            # Calcul de la Walking Speed
            com_initial = np.mean(links_positions[0, :9, :2], axis=0)
            com_final = np.mean(links_positions[-1, :9, :2], axis=0)
            distance_fwd = np.linalg.norm(com_final - com_initial)
            total_time = links_positions.shape[0] * timestep
            walking_speed = distance_fwd / total_time

            # Calcul du Cost of Transport
            cot = compute_mechanical_cot(
                    joints_torques=joints_torques,
                    joints_velocities=joints_velocities,
                    links_positions=links_positions,
                    timestep=timestep
                )

            # Stockage avec les nouveaux axes X et Y
            speed_results.append([body_gain, limb_gain, walking_speed])
            cot_results.append([body_gain, limb_gain, cot])

        except Exception as e:
            # Utile pour ne pas crasher si une des 441 simulations a échoué
            print(f"Erreur sur la simulation {i} : {e}")

    speed_results = np.array(speed_results)
    cot_results = np.array(cot_results)

    # --- HEATMAP VITESSE ---
    plt.figure('Gains Sweep: Walking Speed', figsize=(8, 6))
    plot_2d(
        results=speed_results,
        labels=['Body Gain', 'Limb Gain', 'Walking Speed [m/s]'],
        cmap='viridis'
    )
    plt.title('Walking Speed with respect to Body and Limb Gains')

    # --- HEATMAP CoT ---
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
    """Video of the optimal oscillator amplitudes found in the 3b sweep"""
    folder = f'./logs/ex3b_{label}/'
    os.makedirs(folder, exist_ok=True)
    simulation(
        sim_parameters=SimulationParameters(
            duration=15,
            timestep=timestep,
            spawn_position=[0, 0, 0.1],
            spawn_orientation=[0, 0, np.pi/2],
            drive=2.6,
            limb_body_phase_offset=0.0,
            position_body_gain=optimal_body_gain,
            position_limb_gain=optimal_limb_gain,
        ),
        arena='land',
        output=f'{folder}sim_0',
        record=True,
        record_path=f'{folder}video_{label}.mp4',
        verbose=True,
    )
    return


if __name__ == '__main__':

    #exercise_3a_coordination(timestep=5e-3)
    #analyze_exercise_3a_results()
    #exercise_3b_coordination(timestep=5e-3)
    #analyze_exercise_3b_results()

    # ----------------------------------------------------------------------------------------
    # For videos : 
    
    #exercise_3_disable_limb_spine_coupling(timestep=5e-3)
    #exercise_3_limb_spine_antiphase(timestep=5e-3)
    #exercise_3b_optimal_video(timestep=5e-3, optimal_body_gain=2.5, optimal_limb_gain=2.2, label='speed_optimal')
    exercise_3b_optimal_video(timestep=5e-3, optimal_body_gain=1.0, optimal_limb_gain=1.0, label='cot_optimal')
