"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super().__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.initial_phases = parameters.initial_phases
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = 2*self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.phase_bias = np.zeros([
            self.n_oscillators,
            self.n_oscillators,
        ])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        # self.feedback_gains_swim = np.zeros(self.n_oscillators)
        # self.feedback_gains_walk = np.zeros(self.n_oscillators)

        # # gains for final motor output
        self.position_body_gain = parameters.position_body_gain
        self.position_limb_gain = parameters.position_limb_gain

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

        girdle_fronts (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 relbow_frontresenting the girdle_frontS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        # Example to get global coordinates of robot links
        girdle_fronts = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        # Example to update the drive
        # self.sim_parameters.drive = ...
        # self.set_frequencies(self.sim_parameters)  # f_i
        # self.set_nominal_amplitudes(self.sim_parameters)  # R_i
        # print("girdle_frontGS: {}".format(girdle_fronts[4, 0]))
        # print("drive: {}".format(self.sim_parameters.drive))

        @staticmethod
        def _get_drive(parameters):
            drive = parameters.drive
            if np.isscalar(drive):
                return float(drive)
            drive = np.asarray(drive)
            return float(drive[0]) if drive.ndim > 0 else float(drive)

    def set_frequencies(self, parameters):
        """Set frequencies"""
        #there is 2 frequencies one for the bpdy and one for the legs
        #shape of the freq: np.zeros(self.n_oscillators)

        ######## code estelle #############
        """ 
        FREQUENCIES  (From paper Ijspeert supplelbow_backentary Table S1)
        Body oscillators  (i = 0..15):  active for 1 < d < 5
           ν_body(d) = 0.2·d + 0.3   [Hz]
        
         Limb oscillators  (i = 16..31): active for 1 < d < 3
           ν_limb(d) = 0.2·d + 0.0   [Hz]
        
         Outside the active range the amplitude is driven to 0 (see
         set_nominal_amplitudes)
        """
        """ FORMER CODE
            for i in range(self.n_oscillators):
                self.freqs[i] = 2
        pylog.error('Coupling weights must be set')
            """
        #drive = parameters.drive
        drive = getattr(parameters,'drive', 2.0) #put 2 as a drive if no drive in parameter
        if np.isscalar(drive):
            d = float(drive)
        else:
            drive = np.asarray(drive)
            d = float(drive.flat[0])

        # --- Body oscillators (indices 0-15) ---
        if 1.0 < d < 5.0:
            nu_body = 0.2 * d + 0.3
        elif d >= 5.0:
            nu_body = 0.2 * 5.0 + 0.3   # saturate at d=5
        else:
            nu_body = 0.0                 # below threshold → silent

        # --- Limb oscillators (indices 16-31) ---
        if 1.0 < d < 3.0:
            nu_limb = 0.2 * d + 0.0
        elif d >= 3.0:
            nu_limb = 0.2 * 3.0 + 0.0   # saturate at d=3
        else:
            nu_limb = 0.0                 # above d=3 limbs are silenced

        self.freqs[:self.n_oscillators_body] = nu_body
        self.freqs[self.n_oscillators_body:] = nu_limb

        

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        #shape fo the coupling weights np.zeros([self.n_oscillators,self.n_oscillators,])
        #pylog.error('Coupling weights must be set')

        ####### estelle code ###########
        # w = 10 everywhere (strong enough to lock phases quickly)
        #
        # Limb oscillators (indices 16-31):
        #   4 limbs × 4 oscillators each:
        #     Forelimb L : 16(girdle+), 17(girdle-), 18(elbow+), 19(elbow-)
        #     Forelimb R : 20(girdle+), 21(girdle-), 22(elbow+), 23(elbow-)
        #     Hindlimb L : 24(girdle+), 25(girdle-), 26(knee+),  27(knee-)
        #     Hindlimb R : 28(girdle+), 29(girdle-), 30(knee+),  31(knee-)
        #
        #   Between limbs:
        #     • diagonal (trot): FL-L ↔ HL-R  and  FL-R ↔ HL-L
        #     • contralateral same girdle:  FL-L ↔ FL-R,  HL-L ↔ HL-R
        #
        #   Limb → body coupling:
        #     • each limb girdle+ couples to the nearest body oscillators
        #       (forelimbs → segirdle_backents 0-1, hindlimbs → segirdle_backents 4-5 of the chain)

        weight_axial_contra = getattr(parameters, 'spine_contra_weight', 10.0) #put 10 if no spine_limb_weight is given in parameters
        weight_axial_ipsi = getattr(parameters, 'spine_ipsi_weight', 10.0) #put 10 if no spine_limb_weight is given in parameters
        weight_inter_limb_contra = getattr(parameters, 'inter_limb_contra_weight', 10.0) #put 10 if no spine_limb_weight is given in parameters
        weight_inter_limb_ipsi = getattr(parameters, 'inter_limb_ipsi_weight', 10.0) #put 10 if no spine_limb_weight is given in parameters
        weight_intra_limb_contra = getattr(parameters, 'intra_limb_contra_weight', 10.0) #put 10 if no spine_limb_weight is given in parameters
        weight_intra_limb_ipsi = getattr(parameters, 'intra_limb_ipsi_weight', 10.0) #put 10 if no spine_limb_weight is given in parameters
        weight_limb_body = getattr(parameters, 'limb_spine_weight', 30.0) #put 10 if no spine_limb_weight is given in parameters

        w = self.coupling_weights
        w[:] = 0.0
        #weight = 10.0 #

        # ----- Axial body chain -----
        n_body = self.n_oscillators_body   # 16
        for k in range(self.n_body_joints):   # k = 0..7
            i_L = 2 * k       # left oscillator of pair k
            i_R = 2 * k + 1   # right oscillator of pair k

            # Contralateral coupling (left ↔ right, same pair)
            w[i_L, i_R] = weight_axial_contra
            w[i_R, i_L] = weight_axial_contra

            # Ipsilateral coupling to next segirdle_backent
            if k < self.n_body_joints - 1:
                i_L_next = 2 * (k + 1)
                i_R_next = 2 * (k + 1) + 1
                w[i_L, i_L_next] = weight_axial_ipsi
                w[i_L_next, i_L] = weight_axial_ipsi
                w[i_R, i_R_next] = weight_axial_ipsi
                w[i_R_next, i_R] = weight_axial_ipsi

        # ----- Limb oscillators -----
        # Limb base indices: FL-L=16, FL-R=20, HL-L=24, HL-R=28
        limb_bases = {
            'FL_L': 16, # front limb left
            'FL_R': 20, # front limb right
            'HL_L': 24, # hind limb left
            'HL_R': 28, # hind limb right
        }

        for base in limb_bases.values():
            girdle_front, girdle_back = base, base + 1       # girdle antagonists, for first leg : 16,17
            elbow_front, elbow_back = base + 2, base + 3   # elbow antagonists, for first leg : 18,19

            # Antagonist pairs (strong anti-phase coupling)
            w[girdle_front, girdle_back] = weight_intra_limb_contra
            w[girdle_back, girdle_front] = weight_intra_limb_contra
            w[elbow_front, elbow_back] = weight_intra_limb_contra
            w[elbow_back, elbow_front] = weight_intra_limb_contra

            # Girdle ↔ elbow/knee coupling (circular limb motion)
            w[girdle_front, elbow_front] = weight_intra_limb_ipsi
            w[elbow_front, girdle_front] = weight_intra_limb_ipsi
            w[girdle_back, elbow_back] = weight_intra_limb_ipsi
            w[elbow_back, girdle_back] = weight_intra_limb_ipsi
            #w[girdle_front, elbow_back] = weight
            #w[elbow_back, girdle_front] = weight
            #w[girdle_back, elbow_front] = weight
            #w[elbow_front, girdle_back] = weight

        # Gait trot, diagonals in phase
        """
        trot_pairs = [
            (limb_bases['FL_L'], limb_bases['HL_R']),
            (limb_bases['FL_R'], limb_bases['HL_L']),
        ]
        """
        contralateral_pairs = [
            (limb_bases['FL_L'], limb_bases['FL_R']),
            (limb_bases['HL_L'], limb_bases['HL_R']),
        ]
        ipsilateral_pairs = [
            (limb_bases['FL_L'], limb_bases['HL_L']),
            (limb_bases['FL_R'], limb_bases['HL_R']),
        ]
        for (a, b) in contralateral_pairs:
            w[a, b] = weight_inter_limb_contra
            w[b, a] = weight_inter_limb_contra

        for (a, b) in ipsilateral_pairs:
            w[a, b] = weight_inter_limb_ipsi
            w[b, a] = weight_inter_limb_ipsi

        # Limb → body coupling (girdle+ only)
        # Forelimbs couple near the head (pair 0-1, body osc 0-3)
        # Hindlimbs couple near the tail (pair 4-5, body osc 8-11)
        
        limb_to_body = [
            (limb_bases['FL_L'], [0, 2, 4, 6, 8]),   # FL-L girdle+ → left body osc 0,2
            (limb_bases['FL_R'], [1, 3, 5, 7, 9]),   # FL-R girdle+ → right body osc 1,3
            (limb_bases['HL_L'], [8, 10, 12, 14]), # HL-L girdle+ → left body osc 8,10
            (limb_bases['HL_R'], [9, 11, 13, 15]), # HL-R girdle+ → right body osc 9,11
        ]
        for (limb_osc, body_oscs) in limb_to_body:
            for b in body_oscs:
                w[limb_osc, b] = weight_limb_body
                #w[b, limb_osc] = weight



    def set_phase_bias(self, parameters):
        """Set phase bias"""
        #shape fo the phase bias np.zeros([self.n_oscillators,self.n_oscillators,])
        #pylog.error('Phase bias must be set')
        ###### code estelle #########
        # These set the *target* phase difference between connected oscillators.
        # The phase equation drives  (φ_j - φ_i) → ψ_ij.
        #
        # Axial body chain:
        #   • Contralateral (L↔R same pair): π  → anti-phase = proper undulation
        #   • Ipsilateral (head→tail):  +2π·phase_lag_body  (travelling wave)
        #     phase_lag_body ≈ 2π/8 ≈ 0.785 rad per pair by default
        #     The reverse direction gets the negative.
        #
        # Limbs:
        #   • Antagonist pair: π (anti-phase)
        #   • Girdle → elbow/knee: π/2 (quarter-cycle ahead → circular motion)
        #   • Trot diagonal pair: 0 (in phase)
        #   • Contralateral same girdle: π (anti-phase)
        #   • Limb → body: 0 (limb locks to body wave) # MIGHT DO THE OPPOSITE TO FIT THE PAPER


        psi = self.phase_bias
        psi[:] = 0.0

        # Default phase lag along the body (can be overridden via SimulationParameters)
        phase_lag = (
            parameters.phase_lag_body
            if (hasattr(parameters, 'phase_lag_body') and parameters.phase_lag_body is not None)
            else 2*np.pi / self.n_body_joints   # ~0.785 rad ≈ π/4
        )
        anti_phase = np.pi
        psi_intra_limb_contra = np.pi
        psi_intra_limb_ipsi = np.pi/2
        psi_limb_body = 0

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
                psi[i_L, i_L_next] = phase_lag        # i leads i_next
                psi[i_L_next, i_L] = -phase_lag
                psi[i_R, i_R_next] = phase_lag
                psi[i_R_next, i_R] = -phase_lag

        # ----- Limb oscillators -----
        limb_bases = {
            'FL_L': 16, # front limb left
            'FL_R': 20, # front limb right
            'HL_L': 24, # hind limb left
            'HL_R': 28, # hind limb right
        }

        for base in limb_bases.values():
            girdle_front, girdle_back = base, base + 1
            elbow_front, elbow_back = base + 2, base + 3


            # Antagonist pairs: anti-phase
            psi[girdle_front, girdle_back] = -psi_intra_limb_contra
            psi[girdle_back, girdle_front] = psi_intra_limb_contra
            psi[elbow_front, elbow_back] = -psi_intra_limb_contra
            psi[elbow_back, elbow_front] = psi_intra_limb_contra

            # Girdle leads elbow by π/2 (forward swing then stance)
            psi[girdle_front, elbow_front] = -psi_intra_limb_ipsi
            psi[elbow_front, girdle_front] = psi_intra_limb_ipsi
            psi[girdle_back, elbow_back] = -psi_intra_limb_ipsi
            psi[elbow_back, girdle_back] = psi_intra_limb_ipsi

        # Inter-limb: trot diagonal → in phase (ψ=0, already zero)
        # Contralateral same girdle → anti-phase
        contra_pairs = [
            (16, 20),   # FL-L ↔ FL-R
            (24, 28),   # HL-L ↔ HL-R
        ]
        ipsi_pairs = [
            (16, 24),   # FL-L ↔ FL-R
            (20, 28),   # HL-L ↔ HL-R
        ]
        for (a, b) in contra_pairs:
            psi[a, b] = np.pi
            psi[b, a] = np.pi
        for (a, b) in ipsi_pairs:
            psi[a, b] = np.pi
            psi[b, a] = np.pi
        
        limb_to_body = [
            (limb_bases['FL_L'], [0, 2, 4, 6]),   # FL-L girdle+ → left body osc 0,2
            (limb_bases['FL_R'], [1, 3, 5, 7]),   # FL-R girdle+ → right body osc 1,3
            (limb_bases['HL_L'], [8, 10, 12, 14]), # HL-L girdle+ → left body osc 8,10
            (limb_bases['HL_R'], [9, 11, 13, 15]), # HL-R girdle+ → right body osc 9,11
        ]
        for (limb_osc, body_oscs) in limb_to_body:
            for b in body_oscs:
                psi[limb_osc, b] = psi_limb_body


        # Limb → body: 0 phase bias (already zero, limb in phase with body)


    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        #shape of the rate np.zeros(self.n_oscillators)
        #pylog.error('Convergence rates must be set')

        #### estelle code #####
        self.rates[:] = getattr(parameters,'rates', 3.0)
        #self.rates[:] = 20.0 # from paper

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        #shape of the nominal amplitude np.zeros(self.n_oscillators)
        #pylog.error('Nominal amplitudes must be set')

        #### estelle code ##########
        #drive = parameters.drive
        drive = getattr(parameters,'drive', 2.0) #put 2 as a drive if no drive in parameter

        if np.isscalar(drive):
            d = float(drive)
        else:
            drive = np.asarray(drive)
            d = float(drive.flat[0])

        # Body
        if 1.0 < d < 5.0:
            R_body = 0.065 * d + 0.196
        elif d >= 5.0:
            R_body = 0.065 * 5.0 + 0.196   # saturate
        else:
            R_body = 0.0

        # Limbs: only active during walking regime
        if 1.0 < d < 3.0:
           # R_limb = 0.131 * d + 0.131
           R_limb = 0.131 * d + 0.131
        else:
            R_limb = 0.0   # silenced during swimming (d≥3) or below threshold

        self.nominal_amplitudes[:self.n_oscillators_body] = R_body*self.position_body_gain # potentiellement boger dans le if pour seulement quand on marche
        self.nominal_amplitudes[self.n_oscillators_body:] = R_limb*self.position_limb_gain # potentiellement boger dans le if pour seulement quand on marche

