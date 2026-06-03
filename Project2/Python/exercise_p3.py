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
   
    # ── Dominant frequency from CPG phase rate ────────────────────────────────
    dphase    = np.diff(phases_ss[:, 0]) / timestep   # rad/s
    dom_freq  = np.nanmean(dphase[dphase > 0]) / (2 * np.pi)

    # ── Summary stats ─────────────────────────────────────────────────────────
    mean_lag_deg  = np.degrees(np.mean(lags))
    total_lag_deg = np.degrees(np.sum(lags))
    wavelength    = 360.0 / abs(mean_lag_deg) if mean_lag_deg != 0 else np.inf
    pattern       = 'TRAVELING WAVE' if abs(mean_lag_deg) > 10 else 'STANDING WAVE'

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
    ax.fill_between(joints, phases_rel, alpha=0.15, color='steelblue')
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
    ax2.axhline(45, color='coral', lw=1.2, linestyle='--',
                label='Ideal swim lag (360°/8 = 45°/joint)')
    ax2.axhline(-45, color='coral', lw=1.2, linestyle='--')
    ax2.set_ylabel('Phase lag [°]')
    ax2.set_xlabel('Joint pair  (head → tail)')
    ax2.set_title('B — Inter-joint phase lags  (positive = upstream leads downstream)')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    # Annotate bars
    for bar, val in zip(bars, lags_deg):
        ypos = val + (3 if val >= 0 else -10)
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f'{val:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Summary text box
    summary = (
        f'Mean lag / joint : {mean_lag_deg:.1f}°\n'
        f'Total lag J0→J7  : {total_lag_deg:.1f}°\n'
        f'Wavelength       : {wavelength:.1f} segments\n'
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
    pylog.info(f'Saved: {fname}')
    plt.show()

    # Print clear answer to the assignment question
    pylog.info(
        f'\n{"="*60}\n'
        f'  PHASE LAGS ALONG THE SPINE — {mode_label.upper()}\n'
        f'{"="*60}\n'
        f'  Drive            : {drive_val}\n'
        f'  Per-joint lags   : {np.round(lags_deg, 1)} °\n'
        f'  Mean lag/joint   : {mean_lag_deg:.1f}°  ({np.radians(mean_lag_deg):.3f} rad)\n'
        f'  Total lag J0→J7  : {total_lag_deg:.1f}°  ({np.radians(total_lag_deg):.3f} rad)\n'
        f'  Wavelength       : {wavelength:.1f} body segments\n'
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
    from matplotlib.patches import Patch

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
    pylog.info(f'Saved: {fname}')
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

    results = {}

    for drive_val, arena, label, duration in configs:
        pylog.info(f'Running {label} simulation (drive={drive_val}, arena={arena})')
        n_iter = int(duration / timestep)

        # Build the drive array with exactly n_iter points
        if label == 'ramp':
            drive_input = np.linspace(0, 6, n_iter)
        else:
            drive_input = drive_val   # scalar, fine

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

        sim, sim_data = simulation(          # ← unpack the tuple
            sim_parameters=sim_parameters,
            arena=arena,
            fast=True,
            headless=True,
            output=f'./logs/ex3_1/{label}/',
        )

        state_array = np.array(sim_data.state.array[:])
        phases_debug = state_array[:, :32]
        amps_debug   = state_array[:, 32:64]



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
        """
        fft = np.abs(np.fft.rfft(steady_pos[:, 0]))
        freqs_fft = np.fft.rfftfreq(len(steady_pos), d=timestep)
        dom_freq = freqs_fft[np.argmax(fft[1:]) + 1]
        """

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
        drive_for_plot = drive_input if label == 'ramp' else np.full(len(times), drive_val)

        #plot_spine_analysis(steady_times, steady_pos, lags, drive_val, label)
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

    return results
    #########################################################################
    
def plot_spine_analysis_paper_style(times, state_array, drive_val, mode_label, timestep):
    """
    Reproduce Figure 2 style from Ijspeert 2007:
    A - x_i signals from left body oscillators (muscle outputs)
    B - x_i signals from left limb oscillators  
    C - Instantaneous frequencies
    D - Drive signal
    """
    drive_label = 'ramp 0→6' if mode_label == 'ramp' else drive_val
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'CPG dynamics — {mode_label}  (drive={drive_label})', fontsize=13)

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

    # Index groups
    body_left  = np.arange(0, 16, 2)   # [0,2,4,6,8,10,12,14]
    limb_oscs  = np.arange(16, 32)     # all limb oscillators

    #drive_label = 'ramp 0→6' if mode_label == 'ramp' else drive_val
    #fig.suptitle(f'CPG dynamics — {mode_label}  (drive={drive_label})', fontsize=13)
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
    # Scale bar (π/3 like in paper)
    #ax.plot([times[-1]-1, times[-1]-1], [0, np.pi/3], 'k-', lw=2)
    #ax.text(times[-1]-0.8, np.pi/6, 'π/3', fontsize=8)

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
    #ax.plot([times[-1]-1, times[-1]-1], [0, np.pi/3], 'k-', lw=2)
    #ax.text(times[-1]-0.8, np.pi/6, 'π/3', fontsize=8)

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
    pylog.info(f'Saved: {fname}')
    plt.show()

    

def _plot_coupling_comparison(results, timestep):
    """
    2×2 comparison figure:
      Top row    — body oscillator traces (3 cycles, steady state)
      Bottom row — inter-joint phase lag bar chart
    One column per case (coupled / decoupled).
    """
    
    keys      = ['coupled', 'decoupled']
    
    """
    body_left = np.arange(0, 16, 2)
    colors    = plt.cm.viridis(np.linspace(0, 1, 8))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
    fig.suptitle('Walking: with vs without limb-spine coupling', fontsize=13)

    for col, key in enumerate(keys):
        sa    = results[key]['state_array']
        label = results[key]['label']
        speed = results[key]['speed']

        phases = sa[:, :32]
        amps   = sa[:, 32:64]
        x      = amps * (1 + np.cos(phases))
        n      = x.shape[0]
        half   = n // 2
        times  = np.arange(n) * timestep

        # Period estimate
        dphase = np.diff(phases[half:, 0]) / timestep
        omega  = np.nanmean(dphase[dphase > 0])
        period = 2 * np.pi / omega if omega > 0 else 1.0
        n_show = min(int(3 * period / timestep), n - half)

        # ── Top: stacked body oscillator traces ───────────────────────────
        ax = axes[0, col]
        for idx, osc_i in enumerate(body_left):
            offset = (7 - idx) * 0.8
            ax.plot(times[half:half + n_show],
                    x[half:half + n_show, osc_i] + offset,
                    color=colors[idx], lw=0.9)
        ax.set_title(f'{label}\nspeed = {speed:.4f} m/s', fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel('Time [s]')
        if col == 0:
            ax.set_ylabel('x Body oscillators  (head → tail)')

        # ── Bottom: inter-joint phase lags ────────────────────────────────
        ax2 = axes[1, col]
        ph   = phases[half:, body_left]
        lags = [np.mean(np.arctan2(np.sin(ph[:, k] - ph[:, k+1]),
                                    np.cos(ph[:, k] - ph[:, k+1])))
                for k in range(7)]
        lags_deg = np.degrees(lags)
        bc = ['steelblue' if v >= 0 else 'tomato' for v in lags_deg]
        ax2.bar([f'J{k}→{k+1}' for k in range(7)], lags_deg,
                color=bc, edgecolor='k', linewidth=0.5)
        ax2.axhline(0, color='black', lw=0.8)
        ax2.axhline(45, color='coral', lw=1.1, linestyle='--',
                    label='Swim lag (45°)')
        ax2.set_xlabel('Joint pair')
        ax2.tick_params(axis='x', labelsize=7)
        ax2.legend(fontsize=7)
        if col == 0:
            ax2.set_ylabel('Phase lag [°]')

    plt.tight_layout()
    os.makedirs('./logs/ex3_2/', exist_ok=True)
    plt.savefig('./logs/ex3_2/coupling_comparison.png', dpi=150)
    pylog.info('Saved: logs/ex3_2/coupling_comparison.png')
    plt.show()
    """
    # ── Speed comparison bar chart ────────────────────────────────────────────
    speeds = [results[k]['speed'] for k in keys]
    labels = ['With coupling', 'No coupling']
    bar_colors = ['steelblue', 'tomato']

    _, ax_spd = plt.subplots(figsize=(5, 4))
    bars = ax_spd.bar(labels, speeds, color=bar_colors, edgecolor='k', width=0.4)
    for bar, v in zip(bars, speeds):
        ax_spd.text(bar.get_x() + bar.get_width() / 2,
                    v + max(speeds) * 0.02,
                    f'{v:.4f} m/s', ha='center', va='bottom', fontsize=10)
    ax_spd.set_ylabel('Forward speed [m/s]')
    ax_spd.set_title('Walking speed: limb-spine coupling effect')
    ax_spd.set_ylim(0, max(speeds) * 1.25)
    ax_spd.spines['top'].set_visible(False)
    ax_spd.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('./logs/ex3_2/speed_comparison.png', dpi=150)
    pylog.info('Saved: logs/ex3_2/speed_comparison.png')
    plt.show()


def exercise_3_disable_limb_spine_coupling(timestep):
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
        pylog.info(f'Running: {label}')

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
        from farms_amphibious.data.data import AmphibiousExperimentData
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
        )

        pylog.info(f'  [{label}] forward speed = {fwd_speed:.4f} m/s')

        # ── Per-case plots ────────────────────────────────────────────────
        drive_vec = np.full(n_iter, drive)

        plot_spine_analysis_paper_style(
            times=times,
            state_array=state_array,
            drive_val=drive_vec,
            mode_label=f'walk_{key}',
            timestep=timestep,
        )
        plot_emg_style(
            state_array=state_array,
            drive_val=drive,
            mode_label=f'walk_{key}',
            timestep=timestep,
        )
        """
        plot_phase_lags_analysis(
            state_array=state_array,
            drive_val=drive,
            mode_label=f'walk_{key}',
            timestep=timestep,
        )
        """

    # ── Side-by-side comparison ───────────────────────────────────────────
    _plot_coupling_comparison(results, timestep)
    return results


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

