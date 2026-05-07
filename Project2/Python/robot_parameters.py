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
        # self.position_body_gain = parameters.position_body_gain
        # self.position_limb_gain = parameters.position_limb_gain

        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # psi_ij
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def step(self, time, iteration, salamandra_data):
        """Step function called at each iteration

        Parameters
        ----------

        salamanra_data: salamandra_simulation/data.py::SalamandraData
            Contains the robot data, including network and sensors.

        gps (within the method): Numpy array of shape [9x3]
            Numpy array of size 9x3 representing the GPS positions of each link
            of the robot along the body. The first index [0-8] coressponds to
            the link number from head to tail, and the second index [0,1,2]
            coressponds to the XYZ axis in world coordinate.

        """
        # Example to get global coordinates of robot links
        gps = np.array(
            salamandra_data.sensors.links.urdf_positions()[iteration, :9],
        )
        # Example to update the drive
        # self.sim_parameters.drive = ...
        # self.set_frequencies(self.sim_parameters)  # f_i
        # self.set_nominal_amplitudes(self.sim_parameters)  # R_i
        # print("GPGS: {}".format(gps[4, 0]))
        # print("drive: {}".format(self.sim_parameters.drive))

    def set_frequencies(self, parameters):
        """Set frequencies"""
        #pylog.error('Coupling weights must be set')

        # freq = taille 32 (nombre oscillateurs), 16 premiers les oscillateurs du corps, 16 suivants les oscillateurs des pattes

        for i in range(self.n_oscillators_body):
            self.freqs[i] = parameters.freqs[0] 
        
        for i in range(self.n_oscillators_body, self.n_oscillators):
            self.freqs[i] = parameters.freqs[1] 


    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        pylog.error('Coupling weights must be set')

        

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        pylog.error('Phase bias must be set')

        self.phase_bias[:] = 0.0
        
        if parameters.phase_lag_body is not None:
            phase_lag = parameters.phase_lag_body
        else:
            phase_lag = 2 * np.pi / self.n_body_joints
        
        for i in range(self.n_oscillators_body - 1):
            self.phase_bias[i, i + 1] = phase_lag
            self.phase_bias[i + 1, i] = -phase_lag
        
        phase_lag_limb = getattr(parameters, 'phase_lag_limb', np.pi)
        
        if self.n_oscillators_legs >= 4:
            for i in range(2):
                idx_left = self.n_oscillators_body + i
                idx_right = self.n_oscillators_body + 2 + i
                self.phase_bias[idx_left, idx_right] = phase_lag_limb
                self.phase_bias[idx_right, idx_left] = -phase_lag_limb




    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        #pylog.error('Convergence rates must be set')

        rate_body = getattr(parameters, 'rate_body', 20.0)
        rate_limb = getattr(parameters, 'rate_limb', 20.0)
        
        # Oscillateurs du corps
        self.rates[:self.n_oscillators_body] = rate_body
        
        # Oscillateurs des pattes
        self.rates[self.n_oscillators_body:] = rate_limb

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""

        #pylog.error('Nominal amplitudes must be set')
        
        amp_body = getattr(parameters, 'amp_body', 
                          getattr(parameters, 'amplitude_body', 0.2))
        amp_limb = getattr(parameters, 'amp_limb', 
                          getattr(parameters, 'amplitude_limb', 0.3))
        
        # Oscillateurs du corps
        self.nominal_amplitudes[:self.n_oscillators_body] = amp_body
        
        # Oscillateurs des pattes
        self.nominal_amplitudes[self.n_oscillators_body:] = amp_limb
