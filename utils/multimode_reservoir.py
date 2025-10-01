import numpy as np
import matplotlib.pyplot as plt
from utils.waveguide_main_genetic import executeCore
import utils.utils_functions as ut
import scipy.linalg as sla
from tqdm import tqdm

class MultiModeReservoir:
    def __init__(self,N,n,input_enc='phase',type_=0,lambda_central=1560e-9,steps_=500,delta_amp = 1e-18,seed = 42):
        np.random.seed(seed)
        
        self.obs_size = int(0.5*n*(n+1)) # Size of the observables vector (measuring x-quadratures only)
        if input_enc == 'phase':
            self.alpha = 2.*np.random.rand(N)-1.
            self.beta = 2.*np.random.rand(N)-1.
            self.a_N = (np.exp(-np.linspace(-1,1,N)**2))
        elif input_enc == 'amplitude':
            self.alpha = 0.5*np.random.rand(N)
            self.beta = 0.5*np.ones(N,dtype=float)
            self.delta_N = np.zeros(N,dtype=float)
        if type_ == 0:
            self.pump_polarization = "V"
            self.signal_polarization = "V"
            self.idler_polarization = "V"
            self.window = 200e-9
            self.len_mat = 130
            self.length = 15e-3
            self.pump_width = 2e-9
        if type_ == 2:
            self.pump_polarization = "H"
            self.signal_polarization = "H"
            self.idler_polarization = "V"
            self.window = 30e-9
            self.len_mat = 300
            self.length = 3e-3
            self.pump_width = 1.8e-9
        
        # Define the function to get amplitude and phase from input
        def get_amplitude_phase(input, feedback=None):
            if input_enc == 'phase':
                amplitude = self.a_N
                phase = delta_amp * np.pi * (self.alpha * input + self.beta)
                if feedback is not None:
                    phase += delta_amp*feedback
            elif input_enc == 'amplitude':
                amplitude = self.alpha * input + self.beta
                phase = self.delta_N
                if feedback is not None:
                    amplitude += feedback
            return amplitude.tolist(), phase.tolist()
        self.get_amplitude_phase = get_amplitude_phase
        
        # Compute new random encoding parameters (alpha, beta)
        def redo_alpha_beta():
            if input_enc == 'phase':
                self.alpha = 2.*np.random.rand(N)-1.
                self.beta = 2.*np.random.rand(N)-1.
            elif input_enc == 'amplitude':
                self.alpha = .5*np.random.rand(N)
        self.redo_alpha_beta = redo_alpha_beta

        # Get mask matrix
        self.mask_matr = np.random.rand(N,n)
        def get_mask(scale=1.):
            mask = self.mask_matr #-1. # Mask matrix
            svdmask = sla.svdvals(mask) # Singular values of the mask matrix
            mask *= 1/np.max(abs(svdmask)) # Normalize the mask matrix
            mask *= scale # Multiply the mask matrix by the scaling factor m
            return mask
        self.mask = get_mask
        
        # Redo the mask matrix if the initial one is not good
        def redo_mask(scale=1.):
            self.mask_matr = np.random.rand(N,n)
            return get_mask(scale=scale)
        self.redo_mask = redo_mask
        
        # Define the function to execute the core simulation
        def exeCore(amplitude, phase):
            result = executeCore(length=self.length, type_=str(type_), pm_type = "sinc", lambda_central=lambda_central,
                pump_polarization=self.pump_polarization, window=self.window, signal_polarization=self.signal_polarization, idler_polarization=self.idler_polarization, steps=steps_,
                pump_width=self.pump_width,pump_type = 'custom', process='PDC',
                a_n=amplitude, delta_n=phase, width=3e-6, height=3e-6, crystal_type='KTP')
            return result
        self.exeCore = exeCore
        
        # Define the function to plot modes
        def plot_modes(input):
            amplitude,phase = self.get_amplitude_phase(input)
            result = exeCore(amplitude,phase)
            modes = [(result.signal_temporal_modes[:, i]) for i in range(n)]
            for mode in range(min(n,5)):
                plt.plot(modes[mode][steps_//2-self.len_mat//2:steps_//2+self.len_mat//2], label=f'mode {mode}')
            plt.legend()
            plt.title('output supermodes and measurement frexels')
            inte = np.floor(np.linspace(0, self.len_mat, n + 1)).astype(int)
            for a, i in enumerate(inte):
                plt.vlines(i - 1, 0, np.max(abs(modes[0])), 'black', label=f'frexel {a + 1}')
            plt.xlabel('Wavelength (a.u.)')
            plt.show()
        self.plot_modes = plot_modes
        
        def calculate_covariance_matrix(input,feedback=None):
            amplitude, phase = get_amplitude_phase(input, feedback=feedback) 
            results = executeCore(length=self.length, type_=str(type_), pm_type = "sinc", lambda_central=lambda_central,
                pump_polarization=self.pump_polarization, window=self.window, signal_polarization=self.signal_polarization, idler_polarization=self.idler_polarization, steps=steps_,
                pump_width=self.pump_width,pump_type = 'custom', process='PDC',
                a_n=amplitude, delta_n=phase, width=3e-6, height=3e-6, crystal_type='KTP')
            modes = [(results.signal_temporal_modes[:, i]) for i in range(n)]
                
            lambdas = ut.expand_lambda(results.schmidt_coeffs[0:n])
            inte = np.floor(np.linspace(0, self.len_mat, n + 1)).astype(int)
            U = ut.change_of_basis(n, modes, inte, steps_, self.len_mat)
            for i in range(n):
                U[i,:] = U[i,:] / np.linalg.norm(U[i,:])
            U = ut.expand_matrix(U)
            sigma_diag = np.diag(lambdas)
            sigma_frexel= U.T @ sigma_diag @ U
            return sigma_frexel
        self.covariance_matrix = calculate_covariance_matrix
        
        # Define the function to get reservoir state where u is the input temporal sequence
        def get_reservoir_sequence(u, mask_scale=1., feedback_0=None):
            triu_index = np.triu_indices(n) #indexes of the stored observables
            if feedback_0 is not None:
                feedback = feedback_0
            else:
                feedback = 0.

            mask = self.mask(scale=mask_scale) # Mask matrix
            X = np.zeros((len(u), self.obs_size)) # Matrix to store observables
            for i in tqdm(range(len(u)), desc='Computing reservoir states'):
                cov = self.covariance_matrix(u[i], feedback=feedback) # full covariance matrix
                cov_x = cov[0:2*n:2,0:2*n:2] # x-quadrature covariance matrix
                
                cx_diag = np.diag(cov_x) # Diagonal elements (variances of x-quadratures) for feedback
                feedback = np.sum(mask*cx_diag,axis=1) # Feedback for next timestep

                X[i,:] = cov_x[triu_index] # Store only the upper triangular part of the covariance matrix
            
            return X, feedback
        self.get_reservoir_sequence = get_reservoir_sequence
        
        # Test fading memory
        def test_fading_memory(input, static_steps=20, wash_steps=10, mask_scale=1.):
            u_wash = 2*np.random.rand(wash_steps)-1.
            u_static = input*np.ones(static_steps)
            
            print("Washing out initial conditions...")
            _,feedback = self.get_reservoir_sequence(u_wash, mask_scale=mask_scale)
            print("Initial conditions washed out.")

            print("Injecting static input...")
            # we have to include the initial feedback vector feedback_0
            X, _ = self.get_reservoir_sequence(u_static, mask_scale=mask_scale, feedback_0=feedback)

            for i in range(min(6,X.shape[1])):
                plt.plot(X[:,i],'o-', label=f'obs. {i+1}')
            plt.legend()
            plt.title('Fading memory test: reservoir states for static input')
            plt.xlabel('Timestep')
            plt.ylabel('Reservoir state value')

            plt.show()
        self.test_fading_memory = test_fading_memory