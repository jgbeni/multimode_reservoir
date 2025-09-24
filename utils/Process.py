#
# Copyright 2018
# Integrated Quantum Optics group, Paderborn University
# All rights reserved.
#
# author: Benjamin Brecht
# mail: benjamin.brecht@uni-paderborn.de
#

import utils.Crystal as Crystal
import utils.Phasematching as Phasematching
import utils.Pump as Pump
from numpy import inf, linspace, meshgrid, sqrt, mod, real
from scipy.linalg import svd
import math as math
import matplotlib.pyplot as plt


class Container(object):
    def __init__(self):
        pass


class ParametricProcess(object):
    """ Provides simulation of nonlinear optical processes

    === Public Methods ===
    set_parameters -- sets the calculation parameters
    calculate -- starts the desired calculation of the process
    rebin -- method for rescaling the calculated matrices; this
             can be used when the calculation gridsize prevents
             convenient plotting of the data

    === Private Methods ===
    _calculate_wavelengths -- generates the signal and idler wavelength
                              matrices and vectors for the calculations
                              and later plotting
    _calculate_crystal -- generates the Sellmeier equations for the
                          three fields
    _calculate_phasematching -- generates the phasematching function
                                of the process
    _calculate_pump -- generates the pump function of the process
    _calculate_jsa -- generates the joint spectral amplitude of the process
    _calculate_svd -- generates the singular value decomposition of the
                      joint spectral amplitude

    === Public Variables ===
    settings -- a dictionary containing the calculation parameters
    results -- a container object containing the calculation results
    sol -- the speed of light
    crystal -- an instance of the Crystal class
    phasematching -- an instance of the Phasematching class
    pump -- and instance of the Pump class

    === Private Variables ===
    _WL -- internal flag which indicates whether _calculate_wavelengths has
           been called
    _CR -- internal flag which indicates whether _calculate_crystal has
           been called
    _PM -- internal flag which indicates whether _calculate_phasematching has
           been called
    _PU -- internal flag which indicates whether _calculate_pump has
           been called
    _JSA -- internal flag which indicates whether _calculate_jsa has
           been called
    _msg -- generic internal name for all messages
    _signal -- the signal wavelength range
    _idler -- the idler wavelength range
    _indices -- tuple containing the two refractive indices for h and
                v polarization
    _parser -- dictionary used to parse the polarization directions of
               the fields
    _pump_index -- the refractive index of the pump field
    _signal_index -- the refractive index of the signal field
    _idler_index -- the refractive index of the idler field
    _ngroup_pump -- the group refractive index of the pump field
    _ngroup_signal -- the group refractive index of the signal field
    _ngroup_idler -- the group refractive index of the idler field
    _jsa -- the un-normalized joint spectral amplitude
    _dsig -- signal step width, required for proper normalization
    _did -- idler step width, required for proper normalization
    _norm -- normalization factor of the joint spectral amplitude
    _umat -- matrix from the singular value decomposition which contains
             the idler temporal modes
    _dvec -- vector from the singular value decomposition which contains
             the singular values
    _vmat -- matrix from the singular value decomposition which contains
             the signal temporal modes
    _factor -- rescaling factor for the calculated matrices
    _shape -- internal dumpster for the shape of the matrices to be
              rescaled
    """
    def __init__(self):
        """ Initialise a parametric process with default values

        === @parameters ===
        crystal -- ['KTP', 'LN', 'BBO', 'uniaxial', 'uniaxial_temp']
        temperature -- sample temperature in [degree C]
        hfile -- path to horizontal refractive index file
        vfile -- path to vertical refractive index file
        crystal_cut -- ['X', 'Y', 'Z']
        propagation -- ['X', 'Y', 'Z'] (must not be crystal_cut)
        pump_polarization -- ['H', 'V']
        signal_polarization -- ['H', 'V']
        idler_polarization -- ['H', 'V']
        wg_height -- metallic waveguide height/depth in [m]
        wg_width -- metallic waveguide width in [m]
        grating -- poling period in [m]
        length -- sample length in [m]
        pm_type -- ['gauss', 'sinc', 'mix']
        mixing -- mixing parameter for 'mix'-type phasematching
        pump_index -- refractive index of the pump field <func>
        signal_index -- refractive index of the signal field <func>
        idler_index -- refractive index of the idler field <func>
        signal_center -- central signal wavelength in [m]
        idler_center -- central idler wavelength in [m]
        mismatch -- wavevector mismatch (automatically calculated)
        pump_center -- central pump wavelength in [m]
        pump_wavelength -- matrix containing corresponding pump wavelengths
                           in the signal-idler plane (automatically calculated)
        pump_width -- pump intensity FWHM in [m]
        offset -- offset from central pump wavelength [m]
        signal_wavelength -- matrix containing the signal wavelengths
                             (automatically calculated)
        idler_wavelength -- matrix containing the idler wavelengths
                            (automatically calculated)
        pump_type -- ['normal', 'filtered', 'custom']
        process -- ['PDC', 'BWPDC', 'SFG', 'DFG']
        pump_delay -- pump pulse delay with respect to a reference in [s]
        pump_chirp -- quadratic pump chirp parameters in [s**2]
        pump_temporal_mode -- order of the pump Hermite-Gaussian function
        pump_filter_width -- filter intensity FWHM in [m]
        signal_start -- start wavelength of the signal range in [m]
        signal_stop -- stop wavelength of the signal range in [m]
        signal_steps -- number of gridpoints in signal direction
        idler_start -- start wavelength of the idler range in [m]
        idler_stop -- stop wavelength of the idler range in [m]
        idler_steps -- number of gridpoints in idler direction
        signal_
        
        width -- index of the signal spatial mode in width
        signal_mode_height -- index of the signal spatial mode in height/depth
        idler_mode_width -- index of the idler spatial mode in width
        idler_mode_height -- index of the idler spatial mode in height/depth
        pump_mode_width -- index of the pump spatial mode in width
        pump_mode_height -- index of the pump spatial mode in height/depth
        theta -- crystal angle (used only for 'BBO')
        """
        self.settings = dict(
            # Settings associated with the Crystal class
            crystal='KTP',
            temperature=25,
            hfile=None,
            vfile=None,
            crystal_cut="Z",
            propagation="X",
            pump_polarization="V",
            signal_polarization="V",
            idler_polarization="V",
            wg_height=4E-6,
            wg_width=6E-6,
            # Settings associated with the Phasematching class
            grating=inf,
            length=10E-3,
            pm_type='gauss',
            mixing=0,
            pump_index=None,
            signal_index=None,
            idler_index=None,
            signal_center=800E-9,
            idler_center=800E-9,
            mismatch=None,
            spacer=0,
            reps=1,
            # Settings associated with the Pump class
            pump_center=400E-9,
            pump_wavelength=None,
            pump_width=1E-9,
            offset=0,
            signal_wavelength=None,
            idler_wavelength=None,
            pump_type='normal',
            a_n=[1],
            delta_n=[0],
            process='PDC',
            pump_delay=0,
            pump_chirp=0,
            pump_temporal_mode=0,
            pump_filter_width=None,
            # Other settings
            signal_start=790E-9,
            signal_stop=810E-9,
            signal_steps=250,
            idler_start=790E-9,
            idler_stop=810E-9,
            idler_steps=250,
            signal_mode_width=0,
            signal_mode_height=0,
            idler_mode_width=0,
            idler_mode_height=0,
            pump_mode_width=0,
            pump_mode_height=0,
            theta=0)
        self.sol = 299792458.0
        self.results = Container()
        self._WL = False
        self._CR = False
        self._PM = False
        self._PU = False
        self._JSA = False
        

    def set_parameters(self, **kwargs):
        """ Set the calculation parameters

        Accepts a number of keyword arguments and passes them
        on to the 'settings' dictionary. Sets all calculation
        flags to 'False', enforcing recalculation of all functions.
        Also calculates the idler central wavelength corresponding
        to the given pump and signal wavelengths and the process.
        """
        self._WL = False
        self._CR = False
        self._PM = False
        self._PU = False
        self._JSA = False
        for kwarg in kwargs:
            if self.settings[kwarg] != kwargs[kwarg]:
                self.settings[kwarg] = kwargs[kwarg]
        if self.settings['process'].upper() in ['PDC', 'BWPDC']:
            self.settings['idler_center'] = 1.0 /\
                (1.0 / self.settings['pump_center'] -
                 1.0 / self.settings['signal_center'])
        elif self.settings['process'].upper() == 'SFG':
            self.settings['idler_center'] = 1.0 /\
                (1.0 / self.settings['pump_center'] +
                 1.0 / self.settings['signal_center'])
        elif self.settings['process'].upper() == 'DFG':
            if self.settings['pump_center'] > self.settings['signal_center']:
                self.settings['idler_center'] = 1.0 /\
                    (1.0 / self.settings['signal_center'] -
                     1.0 / self.settings['pump_center'])
            else:
                self.settings['idler_center'] = 1.0 /\
                    (1.0 / self.settings['pump_center'] -
                     1.0 / self.settings['signal_center'])
        _msg = (
            "Calculated idler center wavelength. "
            "Now using pump @ %.2fnm, signal @ %.2fnm, "
            "and idler @ %.2fnm."
            % (self.settings['pump_center'] * 1E9,
               self.settings['signal_center'] * 1E9,
               self.settings['idler_center'] * 1E9))
        #print(_msg)

    def _calculate_wavelengths(self):
        """ Calculate the signal and idler wavelengths

        Generates the signal and idler meshgrid for the matrix
        calculations as well as 1-dimensional ranges for
        plotting and stores the outcomes in the 'results'.
        """
        # Calculate the signal and idler wavelength matrices
        _signal = linspace(self.settings['signal_start'],
                           self.settings['signal_stop'],
                           self.settings['signal_steps'])
        _idler = linspace(self.settings['idler_start'],
                          self.settings['idler_stop'],
                          self.settings['idler_steps'])
        self.settings['signal_wavelength'],\
            self.settings['idler_wavelength'] = meshgrid(_signal, _idler)
        self.results.signal = self.settings['signal_wavelength']
        self.results.idler = self.settings['idler_wavelength']
        self.results.signal_range = self.results.signal[0, :]
        self.results.idler_range = self.results.idler[:, 0]
        self._WL = True

    def _calculate_crystal(self):
        """ Calculate the Sellmeier equations

        Generates the Sellmeier equations for the three fields
        and distributes them accordingly. Also calculates the
        refractive indices and group refractive indices at the
        respective central wavelengths and stores them in the
        'results'.
        """
        try:
            self.crystal.__del__()
        except:
            pass
        self.crystal = Crystal.Crystal(
            self.settings['crystal'],
            self.settings['temperature'],
            self.settings['hfile'],
            self.settings['vfile'],
            self.settings['crystal_cut'],
            self.settings['propagation'],
            self.settings['pump_polarization'],
            self.settings['signal_polarization'],
            self.settings['idler_polarization'],
            self.settings['wg_width'],
            self.settings['wg_height'])
        # Calculate and distribute the indices
        _indices = self.crystal.generate_sellmeier()
        _parser = {'H': 0, 'V': 1}
        if self.settings['crystal'].upper() in ['LN', 'PPLN', 'KTP', 'PPKTP']:
            _pump_index = _indices[_parser[
                self.settings['pump_polarization'].upper()]]
            self.settings['pump_index'] = lambda wl:\
                _pump_index(wl, self.settings['pump_mode_width'],
                            self.settings['pump_mode_height'])
            _signal_index = _indices[_parser[
                self.settings['signal_polarization'].upper()]]
            self.settings['signal_index'] = lambda wl:\
                _signal_index(wl, self.settings['signal_mode_width'],
                              self.settings['signal_mode_height'])
            _idler_index = _indices[_parser[
                self.settings['idler_polarization'].upper()]]
            self.settings['idler_index'] = lambda wl:\
                _idler_index(wl, self.settings['idler_mode_width'],
                             self.settings['idler_mode_height'])
        elif self.settings['crystal'].upper() == 'BBO':
            _pump_index = _indices[_parser[
                self.settings['pump_polarization'].upper()]]
            self.settings['pump_index'] = lambda wl: \
                _pump_index(wl, self.settings['theta'])
            _signal_index = _indices[_parser[
                self.settings['signal_polarization'].upper()]]
            self.settings['signal_index'] = lambda wl: \
                _signal_index(wl, self.settings['theta'])
            _idler_index = _indices[_parser[
                self.settings['idler_polarization'].upper()]]
            self.settings['idler_index'] = lambda wl: \
                _idler_index(wl, self.settings['theta'])
        else:
            self.settings['pump_index'] = _indices[_parser[
                self.settings['pump_polarization'].upper()]]
            self.settings['signal_index'] = _indices[_parser[
                self.settings['signal_polarization'].upper()]]
            self.settings['idler_index'] = _indices[_parser[
                self.settings['idler_polarization'].upper()]]
        # Refractive indices at central wavelengths
        self.results.npump = self.settings['pump_index'](
            self.settings['pump_center'])
        self.results.nsignal = self.settings['signal_index'](
            self.settings['signal_center'])
        self.results.nidler = self.settings['idler_index'](
            self.settings['idler_center'])
        # Group velocities at central wavelengths
        _ngroup_pump = self.results.npump - self.settings['pump_center'] *\
            (self.settings['pump_index'](self.settings['pump_center'] + 1E-9) -
             self.settings['pump_index'](self.settings['pump_center'])) / 1E-9
        self.results.vpump = self.sol / _ngroup_pump
        _ngroup_signal = self.results.nsignal -\
            self.settings['signal_center'] *\
            (self.settings['signal_index'](
                self.settings['signal_center'] + 1E-9) -
             self.settings['signal_index'](
                self.settings['signal_center'])) / 1E-9
        self.results.vsignal = self.sol / _ngroup_signal
        _ngroup_idler = self.results.nidler -\
            self.settings['idler_center'] *\
            (self.settings['idler_index'](
                self.settings['idler_center'] + 1E-9) -
             self.settings['idler_index'](
                self.settings['idler_center'])) / 1E-9
        self.results.vidler = self.sol / _ngroup_idler
        self._CR = True

    def _calculate_phasematching(self):
        """ Calculate the phasematching function

        Generates the phasematching matrix and stores it in the
        'results'. Also, if 'grating' is denoted 'inf', it calculates
        the ideal poling period for quasi-phasematching and writes
        it to the settings dictionary.
        """
        if not self._WL:
            self._calculate_wavelengths()
        if not self._CR:
            self._calculate_crystal()
        # Initialize the Phasematching class
        try:
            self.phasematching.__del__()
            print("Deleting")
        except:
            pass
        self.phasematching = Phasematching.Phasematching(
            self.settings['grating'],
            self.settings['length'],
            self.settings['pm_type'],
            self.settings['mixing'],
            self.settings['process'],
            self.settings['pump_index'],
            self.settings['signal_index'],
            self.settings['idler_index'],
            self.settings['pump_center'],
            self.settings['signal_center'],
            self.settings['idler_center'],
            self.settings['signal_wavelength'],
            self.settings['idler_wavelength'],
            self.settings['mismatch'],
            self.settings['theta'],
            self.settings['spacer'],
            self.settings['reps'])
        self.results.phasematching = self.phasematching.phasematching()
        self.results.poling_period = self.phasematching.grating
        self.settings['grating'] = self.phasematching.grating
        self._mismatch = self.phasematching._mismatch
        self._PM = True

    def _calculate_pump(self):
        """ Calculate the pump function

        Generates the pump matrix and stores it in the 'results'.
        """
        if not self._WL:
            self._calculate_wavelengths()
        try:
            self.pump.__del__()
        except:
            pass
        self.pump = Pump.Pump(
            self.settings['pump_center'],
            self.settings['pump_wavelength'],
            self.settings['pump_width'],
            self.settings['offset'],
            self.settings['signal_wavelength'],
            self.settings['idler_wavelength'],
            self.settings['pump_type'],
            self.settings['process'],
            self.settings['a_n'],
            self.settings['delta_n'],
            self.settings['pump_delay'],
            self.settings['pump_chirp'],
            self.settings['pump_temporal_mode'],
            self.settings['pump_filter_width'])
        self.results.pump = self.pump.pump()
        self.results.pump_interv = self.pump.pump_interv
        self._PU = True

    def _calculate_jsa(self):
        """ Calculate the joint spectral amplitude

        Generates the complex-valued JSA matrix and stores
        a normalized version in the 'results'. Also calculates
        the signal and idler stepwdiths, as well as the signal
        and idler marginal distributions and stores them in the
        'results'.
        """
        if not self._PM:
            self._calculate_phasematching()
        if not self._PU:
            self._calculate_pump()
        
        _jsa = self.results.pump * self.results.phasematching
        _dsig = self.results.signal_range[1] -\
            self.results.signal_range[0]
        _did = self.results.idler_range[1] -\
            self.results.idler_range[0]
        _norm = sqrt((abs(_jsa)**2).sum() * _dsig * _did)
        _jsa /= _norm
        self.results.jsa = _jsa
        self.results.signal_step = _dsig
        self.results.idler_step = _did
        self.results.signal_marginal = _jsa.sum(axis=0)
        self.results.idler_marginal = _jsa.sum(axis=1)
        self._JSA = True


    def _calculate_svd(self):
        """ Calculate the singular value decomposition

        Calculates the singular values decomposition of the
        joint spectral amplitude. Normalizes both, the
        Schmidt coefficients and the Schmidt modes and
        stores them in the 'results'.
        """
        if not self._JSA:
            self._calculate_jsa()
        _dsig = self.results.signal_step
        _did = self.results.idler_step
        _umat, _dvec, _vmat = svd(self.results.jsa)
        _dvec *= sqrt(_dsig * _did)
        _umat /= sqrt(_did)
        _vmat /= sqrt(_dsig)
        self.results.schmidt_coeffs = _dvec
        self.results.signal_temporal_modes = _vmat.T.conjugate()
        self.results.idler_temporal_modes = _umat

    def calculate(self, key):
        """ Wrapper function for the calculations

        Accepts a keyword and runs the corresponding
        calculations. Note that the internal
        _calculate_XX functions automatically call
        the necessary precursor functions.

        @arg key: ['WL', 'WAVELENGTHS', 'CRYSTAL', 'PM',
                   'PHASEMATCHING', 'PUMP', 'JSA', 'SVD']
                   ignores cases
        """
        if key.upper() in ['WL', 'WAVELENGTHS']:
            if not self._WL:
                self._calculate_wavelengths()
        if key.upper() == 'CRYSTAL':
            if not self._CR:
                self._calculate_crystal()
        if key.upper() in ['PM', 'PHASEMATCHING']:
            if not self._PM:
                self._calculate_phasematching()
        if key.upper() == "PUMP":
            if not self._PU:
                self._calculate_pump()
        if key.upper() == "JSA":
            self._calculate_jsa()
        if key.upper() == "SVD":
            self._calculate_svd()

    def rebin(self, factor):
        """ Rescale the calculation results

        If the JSA has not yet been calculated, it is calculated
        first. Then, if the rescaling factor matches with the gridsize,
        the JSA, signal, idler, pump, and phasematching matrices are
        downscaled and stored in the 'results'. This function is intended
        for use with high-res calculations and low-res plotting.
        """
        if not self._JSA:
            self._calculate_jsa()
        _factor = factor
        if not mod(self.settings['signal_steps'], factor) == 0 or\
                not mod(self.settings['idler_steps'], factor) == 0:
            print("Could not rebin due to dimension mismatch.")
        else:
            _shape = self.results.jsa.shape
            self.results.jsa_rebin = self.results.jsa.reshape(
                (_shape[0] // _factor, _factor,
                 _shape[1] // _factor, _factor)).mean(-1).mean(1)
            _shape = self.results.signal.shape
            self.results.signal_rebin = self.results.signal.reshape(
                (_shape[0] // _factor, _factor,
                 _shape[1] // _factor, _factor)).mean(-1).mean(1)
            _shape = self.results.idler.shape
            self.results.idler_rebin = self.results.idler.reshape(
                (_shape[0] // _factor, _factor,
                 _shape[1] // _factor, _factor)).mean(-1).mean(1)
            _shape = self.results.pump.shape
            self.results.pump_rebin = self.results.pump.reshape(
                (_shape[0] // _factor, _factor,
                 _shape[1] // _factor, _factor)).mean(-1).mean(1)
            _shape = self.results.phasematching.shape
            self.results.phasematching_rebin = \
                self.results.phasematching.reshape(
                    (_shape[0] // _factor, _factor,
                     _shape[1] // _factor, _factor)).mean(-1).mean(1)
