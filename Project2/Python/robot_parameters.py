"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super().__init__()

        self.sim_parameters = parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.update_drive = getattr(parameters, 'update_drive', False)
        self.current_gait = 'walk'
        self.target_drive = 2.0
        self.alpha_drive = 0.15
        self.alpha_filter = 0.3
        self.filtered_head = 0.0
        self.filtered_feet = 0.0

        self.transition_counter = 0
        self.debounce_steps = 5
        self.to_swim_threshold = 0.3
        self.to_walk_threshold = 1.2

        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2 * self.n_body_joints
        self.n_oscillators_legs = 2 * self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        # ----- Limb oscillators -----
        # Limb base indices: FL-L=16, FL-R=20, HL-L=24, HL-R=28
        self.limb_bases = {
            'FL_L': 16, # front limb left
            'FL_R': 20, # front limb right
            'HL_L': 24, # hind limb left
            'HL_R': 28, # hind limb right
        }

        self.limb_to_body = [
            (self.limb_bases['FL_L'], [0, 2, 4, 6]),
            (self.limb_bases['FL_R'], [1, 3, 5, 7]),
            (self.limb_bases['HL_L'], [8, 10, 12, 14]),
            (self.limb_bases['HL_R'], [9, 11, 13, 15]),
        ]

        # # gains for final motor output
        self.position_body_gain = getattr(parameters,'position_body_gain', 1.0) 
        self.position_limb_gain = getattr(parameters,'position_limb_gain' , 1.0) 
        
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        #print(f"test: {parameters.test_value}")
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # psi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, time, iteration, salamandra_data):
        """Stelbow_front function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        shoulder_fronts (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 relbow_frontresenting the shoulder_frontS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        # Example to get global coordinates of robot links
        shoulder_fronts = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )

        if getattr(self.sim_parameters, 'drive_ramp', False):
            d_start = getattr(self.sim_parameters, 'drive_ramp_start', 1.0)
            d_end = getattr(self.sim_parameters, 'drive_ramp_end', 5.0)
            duration = getattr(self.sim_parameters, 'duration', 40.0)
            self.sim_parameters.drive = d_start + (d_end - d_start) * min(time / duration, 1.0)
            self.set_frequencies(self.sim_parameters)
            self.set_nominal_amplitudes(self.sim_parameters)
            self.set_phase_bias(self.sim_parameters)
        elif self.update_drive:
            index = 0 if iteration == 0 else (iteration - 1)
            contacts_all = np.linalg.norm(np.array(
                salamandra_data.sensors.contacts.totals()[index]
            ), axis=1)

            contacts_body = contacts_all[:9]
            contacts_feet = contacts_all[10:18:2]

            contact_head = contacts_body[0]
            total_feet = np.sum(contacts_feet > 0.1)
            total_body_rest = np.sum(contacts_body[1:] > 0.1)

            self.filtered_head = self.alpha_filter * contact_head + (1.0 - self.alpha_filter) * self.filtered_head
            self.filtered_feet = self.alpha_filter * total_feet + (1.0 - self.alpha_filter) * self.filtered_feet

            want_swim = (
                self.current_gait == 'walk'
                and self.filtered_feet <= self.to_swim_threshold
                and total_body_rest == 0
                and self.filtered_head < 0.05
            )

            want_walk = (
                self.current_gait == 'swim'
                and (self.filtered_head > 0.05 or self.filtered_feet >= self.to_walk_threshold)
            )

            if want_swim or want_walk:
                self.transition_counter += 1
            else:
                self.transition_counter = 0

            if self.transition_counter >= self.debounce_steps:
                if want_swim:
                    self.current_gait = 'swim'
                    self.target_drive = 4.0
                elif want_walk:
                    self.current_gait = 'walk'
                    self.target_drive = 2.0
                self.transition_counter = 0

            current_drive = getattr(self.sim_parameters, 'drive', 2.0)
            self.sim_parameters.drive = current_drive + self.alpha_drive * (self.target_drive - current_drive)

        drive_raw = getattr(self.sim_parameters, 'drive', 2.0)
        self._drive_array = (
            None if np.isscalar(drive_raw)
            else np.asarray(drive_raw).copy()
        )
            # Update CPG from gait state
        self.set_frequencies(self.sim_parameters)
        self.set_nominal_amplitudes(self.sim_parameters)
        self.set_phase_bias(self.sim_parameters)

    def set_frequencies(self, parameters):
        """Set frequencies"""
        drive = getattr(parameters, 'drive', 2.0)
        if np.isscalar(drive):
            d = float(drive)
        else:
            drive = np.asarray(drive)
            d = float(drive.flat[0])

        # --- Body oscillators (indices 0-15) ---
        if 1.0 < d < 5.0:
            nu_body = 0.2 * d + 0.3
        elif d >= 5.0:
            nu_body = 0
        else:
            nu_body = 0.0                 # below threshold → silent

        # --- Limb oscillators (indices 16-31) ---
        if 1.0 < d < 3.0:
            nu_limb = 0.2 * d + 0.0
        elif d >= 3.0:
            nu_limb = 0
        else:
            nu_limb = 0.0

        self.freqs[:self.n_oscillators_body] = 2 * np.pi * nu_body
        self.freqs[self.n_oscillators_body:] = 2 * np.pi * nu_limb

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        weight_axial_contra = getattr(parameters, 'spine_contra_weight', 10.0)
        weight_axial_ipsi = getattr(parameters, 'spine_ipsi_weight', 10.0)
        weight_inter_limb_contra = getattr(parameters, 'inter_limb_contra_weight', 10.0)
        weight_inter_limb_ipsi = getattr(parameters, 'inter_limb_ipsi_weight', 10.0)
        weight_intra_limb_contra = getattr(parameters, 'intra_limb_contra_weight', 10.0)
        weight_intra_limb_ipsi = getattr(parameters, 'intra_limb_ipsi_weight', 10.0)
        weight_limb_body = getattr(parameters, 'limb_spine_weight', 30.0)
        weight_body_limb = getattr(parameters, 'spine_limb_weight', 10.0)

        w = self.coupling_weights
        w[:] = 0.0

        n_body = self.n_oscillators_body
        for k in range(self.n_body_joints):
            i_L = 2 * k
            i_R = 2 * k + 1

            w[i_L, i_R] = weight_axial_contra
            w[i_R, i_L] = weight_axial_contra

            # Ipsilateral coupling to next seshoulder_backent
            if k < self.n_body_joints - 1:
                i_L_next = 2 * (k + 1)
                i_R_next = 2 * (k + 1) + 1
                w[i_L, i_L_next] = weight_axial_ipsi
                w[i_L_next, i_L] = weight_axial_ipsi
                w[i_R, i_R_next] = weight_axial_ipsi
                w[i_R_next, i_R] = weight_axial_ipsi

        for base in self.limb_bases.values():
            shoulder_front, shoulder_back = base, base + 1
            elbow_front, elbow_back = base + 2, base + 3

            # Antagonist pairs (strong anti-phase coupling)
            w[shoulder_front, shoulder_back] = weight_intra_limb_contra
            w[shoulder_back, shoulder_front] = weight_intra_limb_contra
            w[elbow_front, elbow_back] = weight_intra_limb_contra
            w[elbow_back, elbow_front] = weight_intra_limb_contra

            # shoulder ↔ elbow/knee coupling (circular limb motion)
            w[shoulder_front, elbow_front] = weight_intra_limb_ipsi
            w[elbow_front, shoulder_front] = weight_intra_limb_ipsi
            w[shoulder_back, elbow_back] = weight_intra_limb_ipsi
            w[elbow_back, shoulder_back] = weight_intra_limb_ipsi

        contralateral_pairs = [
            (self.limb_bases['FL_L'], self.limb_bases['FL_R']),
            (self.limb_bases['HL_L'], self.limb_bases['HL_R']),
        ]
        ipsilateral_pairs = [
            (self.limb_bases['FL_L'], self.limb_bases['HL_L']),
            (self.limb_bases['FL_R'], self.limb_bases['HL_R']),
        ]
        trot_pairs = [
            (self.limb_bases['FL_L'], self.limb_bases['HL_R']),
            (self.limb_bases['FL_R'], self.limb_bases['HL_L']),
        ]
        for (a, b) in contralateral_pairs:
            w[a, b] = weight_inter_limb_contra
            w[b, a] = weight_inter_limb_contra

        for (a, b) in ipsilateral_pairs:
            w[a, b] = weight_inter_limb_ipsi
            w[b, a] = weight_inter_limb_ipsi

        for (a, b) in trot_pairs:
            w[a, b] = weight_inter_limb_contra
            w[b, a] = weight_inter_limb_contra

        for (limb_osc, body_oscs) in self.limb_to_body:
            for b in body_oscs:
                w[limb_osc, b] = weight_body_limb
                w[b, limb_osc] = weight_limb_body

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        psi = self.phase_bias
        psi[:] = 0.0

        drive = getattr(parameters, 'drive', 2.0)
        if np.isscalar(drive):
            d = float(drive)
        else:
            drive = np.asarray(drive)
            d = float(drive.flat[0])

        if 1.0 < d < 3.0:
            phase_lag = 0.0
        else:
            phase_lag = (
                parameters.phase_lag_body
                if (hasattr(parameters, 'phase_lag_body') and parameters.phase_lag_body is not None)
                else -2*np.pi / self.n_body_joints   # ~0.785 rad ≈ π/4
            )
        anti_phase = np.pi
        psi_intra_limb_contra = np.pi
        psi_intra_limb_ipsi = np.pi / 2
        psi_limb_body = getattr(parameters, 'limb_body_phase_offset', 0.0)
        # ----- Axial chain -----
        for k in range(self.n_body_joints):
            i_L = 2 * k
            i_R = 2 * k + 1
            
            # positive : Left to right
            # positive : From head to tail

            # Contralateral: anti-phase
            psi[i_L, i_R] = anti_phase
            psi[i_R, i_L] = -anti_phase

            # Ipsilateral: travelling wave (head→tail positive lag)
            if k < self.n_body_joints - 1:
                i_L_next = 2 * (k + 1)
                i_R_next = 2 * (k + 1) + 1
                if 1.0 < d < 3.0 and k == 3:
                    current_lag = np.pi
                else:
                    current_lag = phase_lag

                psi[i_L, i_L_next] = current_lag
                psi[i_L_next, i_L] = -current_lag
                psi[i_R, i_R_next] = current_lag
                psi[i_R_next, i_R] = -current_lag

        for base in self.limb_bases.values():
            shoulder_front, shoulder_back = base, base + 1
            elbow_front, elbow_back = base + 2, base + 3


            # Antagonist pairs: anti-phase
            psi[shoulder_front, shoulder_back] = -psi_intra_limb_contra
            psi[shoulder_back, shoulder_front] = psi_intra_limb_contra
            psi[elbow_front, elbow_back] = -psi_intra_limb_contra
            psi[elbow_back, elbow_front] = psi_intra_limb_contra

            # shoulder leads elbow by π/2 (forward swing then stance)
            psi[shoulder_front, elbow_front] = -psi_intra_limb_ipsi
            psi[elbow_front, shoulder_front] = psi_intra_limb_ipsi
            psi[shoulder_back, elbow_back] = -psi_intra_limb_ipsi
            psi[elbow_back, shoulder_back] = psi_intra_limb_ipsi

        contralateral_pairs = [
            (self.limb_bases['FL_L'], self.limb_bases['FL_R']),
            (self.limb_bases['HL_L'], self.limb_bases['HL_R']),
        ]
        ipsilateral_pairs = [
            (self.limb_bases['FL_L'], self.limb_bases['HL_L']),
            (self.limb_bases['FL_R'], self.limb_bases['HL_R']),
        ]
        trot_pairs = [
            (self.limb_bases['FL_L'], self.limb_bases['HL_R']),
            (self.limb_bases['FL_R'], self.limb_bases['HL_L']),
        ]

        for (a, b) in contralateral_pairs:
            psi[a, b] = np.pi
            psi[b, a] = np.pi

        for (a, b) in ipsilateral_pairs:
            psi[a, b] = np.pi
            psi[b, a] = np.pi

        for (a, b) in trot_pairs:
            psi[a, b] = 0.0
            psi[b, a] = 0.0

        for (limb_osc, body_oscs) in self.limb_to_body:
            for b in body_oscs:
                psi[limb_osc, b] = -psi_limb_body
                psi[b, limb_osc] = psi_limb_body

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates[:] = getattr(parameters, 'rates', 100.0)

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        drive = getattr(parameters, 'drive', 2.0)
        if np.isscalar(drive):
            d = float(drive)
        else:
            drive = np.asarray(drive)
            d = float(drive.flat[0])

        # Body
        if 1.0 < d < 5.0:
            R_body = 0.065 * d + 0.196
        elif d >= 5.0:
            R_body = 0.065 * 5.0 + 0.196
        else:
            R_body = 0.0

        # Limbs: only active during walking regime
        if 1.0 < d < 3.0:
            R_limb = 0.131 * d + 0.131
        else:
            R_limb = 0.0

        self.nominal_amplitudes[:self.n_oscillators_body] = R_body * self.position_body_gain
        self.nominal_amplitudes[self.n_oscillators_body:] = R_limb * self.position_limb_gain