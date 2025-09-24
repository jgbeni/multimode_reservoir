#
# Copyright 2018
# Integrated Quantum Optics group, Paderborn University
# All rights reserved.
#
# author: Benjamin Brecht
# mail: benjamin.brecht@uni-paderborn.de
#

import numpy as np
from scipy.integrate import quad, fixed_quad
from time import time 
import matplotlib.pyplot as plt
from numba import njit, prange
from numba.typed import List


def n(matrix):
    matrix_max = np.max(matrix)
    return matrix / matrix_max

class Phasematching(object):
    """ Provides a phasematching function of a nonlinear process.

    === Public Methods ===
    phasematching -- generates the phasematching function from
                     the provided parameters

    === Private Methods ===
    _sinc -- redefinition of the sinc function
    _kvector -- calculates a wavector
    _grating -- calculates the poling period if none is supplied
    _wavevector_mismatch -- generates a matrix containing the
                            values of the wavevector mismatch in
                            the signal and idler frequency
                            plane
    _g -- g function 
    _sequence -- calculating the z_

    === Public Variables ===
    duty -- duty cycle poling (default: lambda z: 0.5)
    grating -- poling period [m] (default: np.inf)
    length -- waveguide length [m] (default: 10E-3)
    type -- phasematching approximation; must be in ['gauss',
            'sinc', 'mix'] (default: 'gauss')
    mixing -- weighting parameter of gaussian and sinc phasematching;
              mixing = 0 means completely gaussian (default: 0)
    process -- nonlinear process (default: 'PDC')
    pump_index -- pump sellmeier equation (default: None)
    signal_index -- signal sellmeier equation (default: None)
    idler_index -- idler sellmeier equation (default: None)
    pump_center -- pump central wavelength (default: None)
    signal_center -- signal central wavelength (default: None)
    idler_center -- idler central wavelength (default: None)
    signal_wl -- matrix containing signal wavelengths (default: None)
    idler_wl -- matrix containing idler wavelengths (default: None)
    mismatch -- matrix containing the wavevector mismatch (default: None)

    === Private Variables ===
    _sequence -- z_ calculated with the duty cycle
    _result -- about every calculation result
    _kpump -- pump wavevector
    _ksignal -- signal wavevector
    _kidler -- idler wavevector
    _grating -- poling period
    _mismatch -- wavevector mismatch
    """
    def __init__(self,
                 grating=np.inf, length=10E-3, pm_type="gauss", mixing=0,
                 process="PDC", pump_index=None, signal_index=None,
                 idler_index=None, pump_center=None, signal_center=None,
                 idler_center=None, signal_wavelength=None,
                 idler_wavelength=None, mismatch=None, theta=0,
                 spacer=0, reps=1):
        """ Initialise a phasematching with default values. """
        
        self.grating = grating
        self.length = length
        self.type = pm_type
        self.mixing = mixing
        self.process = process
        self.pump_index = pump_index
        self.signal_index = signal_index
        self.idler_index = idler_index
        self.pump_center = pump_center
        self.signal_center = signal_center
        self.idler_center = idler_center
        self.signal_wl = signal_wavelength
        self.idler_wl = idler_wavelength
        self.mismatch = mismatch
        self.spacer = spacer
        self.reps = reps

   
    def _sinc(self, x):
        """ Redefinition of the sinc function. """
        return np.sin(x) / x

    def _kvector(self, wl, index):
        """ Calculates a wavevector from wavelength and refractive index.

        === Returns ===
        _result -- the calculated wavevector
        """
        _result = 2 * np.pi * index(wl) / wl
        return _result


    
    def _grating(self):
        """
        Calculates the poling period required for perfect phasematching
        of the supplied fields.

        === Returns ===
        _grating -- poling period
        """
        _kpump = self._kvector(self.pump_center, self.pump_index)
        _ksignal = self._kvector(self.signal_center, self.signal_index)
        _kidler = self._kvector(self.idler_center, self.idler_index)
        if self.process.upper() == 'PDC':
            _mismatch = _kpump - _ksignal - _kidler
        elif self.process.upper() in ['SFG', 'BWPDC']:
            _mismatch = _kpump + _ksignal - _kidler
        elif self.process.upper() == 'DFG':
            _mismatch = _kpump - _ksignal + _kidler
        _grating = 2 * np.pi / _mismatch
        return _grating

    def _delta_k_central(self):
        """
        Calculates the poling period required for perfect phasematching
        of the supplied fields.

        === Returns ===
        _grating -- poling period
        """
        #print('calcul des index pour la fréq centrale')
        _kpump = self._kvector(self.pump_center, self.pump_index)
        _ksignal = self._kvector(self.signal_center, self.signal_index)
        _kidler = self._kvector(self.idler_center, self.idler_index)
        if self.process.upper() == 'PDC':
            _mismatch = _kpump - _ksignal - _kidler
        elif self.process.upper() in ['SFG', 'BWPDC']:
            _mismatch = _kpump + _ksignal - _kidler
        elif self.process.upper() == 'DFG':
            _mismatch = _kpump - _ksignal + _kidler
        return _mismatch

    def _wavevector_mismatch(self):
        """ Calculates the wavevector mismatch matrix.

        === Returns ===
        _mismatch -- matrix containing the wavevector mismatch
                     in signal and idler frequency plane
        """
#         print(f'self.process.up is {self.process.upper()}')
        if self.process.upper() == 'PDC':
            _pump_wl = 1.0 / (1.0 / self.signal_wl + 1.0 / self.idler_wl)
            _mismatch = self._kvector(_pump_wl, self.pump_index) -\
                self._kvector(self.signal_wl, self.signal_index) -\
                self._kvector(self.idler_wl, self.idler_index)  #- 2 * np.pi / self.grating                           
        if self.process.upper() in ['SFG', 'BWPDC']:
            _pump_wl = 1.0 / (1.0 / self.idler_wl - 1.0 / self.signal_wl)
            _mismatch = self._kvector(_pump_wl, self.pump_index) +\
                self._kvector(self.signal_wl, self.signal_index) -\
                self._kvector(self.idler_wl, self.idler_index) -\
                2 * np.pi / self.grating
        if self.process.upper() == 'DFG':
            _pump_wl = 1.0 / (1.0 / self.signal_wl - 1.0 / self.idler_wl)
            _mismatch = self._kvector(_pump_wl, self.pump_index) -\
                self._kvector(self.signal_wl, self.signal_index) +\
                self._kvector(self.idler_wl, self.idler_index) -\
                2 * np.pi / self.grating
        self._mismatch = _mismatch
        return _mismatch    

    def _wavevector_mismatch_bulk(self):
        if self.process.upper() == 'PDC':
            _pump_wl = 1.0 / (1.0 / self.signal_wl + 1.0 / self.idler_wl)
            _mismatch_bulk = self._kvector(_pump_wl, self.pump_index) -\
                self._kvector(self.signal_wl, self.signal_index) -\
                self._kvector(self.idler_wl, self.idler_index)
        if self.process.upper() in ['SFG', 'BWPDC']:
            _pump_wl = 1.0 / (1.0 / self.idler_wl - 1.0 / self.signal_wl)
            _mismatch_bulk = self._kvector(_pump_wl, self.pump_index) +\
                self._kvector(self.signal_wl, self.signal_index) -\
                self._kvector(self.idler_wl, self.idler_index)
        if self.process.upper() == 'DFG':
            _pump_wl = 1.0 / (1.0 / self.signal_wl - 1.0 / self.idler_wl)
            _mismatch_bulk = self._kvector(_pump_wl, self.pump_index) -\
                self._kvector(self.signal_wl, self.signal_index) +\
                self._kvector(self.idler_wl, self.idler_index)
        return _mismatch_bulk
              
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _pm(mismatch, xn, L, delta_c, a, b, c, d):
        rows = len(mismatch)
        cols = len(mismatch[0])  # Assuming all rows have the same length
        result = np.zeros((rows, cols), dtype=np.complex128)  # Initialize with complex dtype
        
        N = len(xn)
        lc=np.pi/(delta_c)
        def duty(x,a,b,c):
            return d+a*x+b*x**2+c*x**3
        zn = np.array([(n - 0.5) * lc * duty(n*L/N,a,b,c) for n in range(-N // 2, N // 2)])
        for i in prange(rows):
            for j in prange(cols):
                d = mismatch[i, j]
                sum_term = 0.0j
                for k in prange(N):
                    sum_term += xn[k] * np.exp(1j * d * zn[k])
                result[i, j] = ((-2j) * (lc / L) * (np.sin(d * lc / 2) / (d * lc / 2)) * sum_term)
        return result

    def phasematching(self):
        """ Calculates the phasematching matrix

        === Returns ===
        _result -- matrix containing the phasematching in signal
                   and idler frequency plane
        """
        if self.grating == np.inf:
            self.grating = self._grating()
        self.mismatch = self._wavevector_mismatch()
        if self.type.upper() == 'GAUSS':
            _result = np.exp(-0.193 * (0.5 * self.length * self.mismatch)**2) *\
                np.exp(1j * 0.5 * self.length * self.mismatch)
        elif self.type.upper() == 'SINC':
            _result = self._sinc(0.5 * self.length * (self.mismatch-2 * np.pi / self.grating))  ####periodic poling approximation
        elif self.type.upper() == 'MIX':
            _result = (self.mixing * self._sinc(0.5 * self.length *
                       self.mismatch) + (1 - self.mixing) *
                       np.exp(-0.193 * (0.5 * self.length * self.mismatch)**2)) *\
                np.exp(1j * 0.5 * self.length * self.mismatch)
        elif self.type.upper() == "SPACER":
            print("SPACER")
            print(self.spacer, self.length)
            print(self.reps)
            self.mismatch_bulk = self._wavevector_mismatch_bulk()
            _phi = self.mismatch * self.length +\
                self.mismatch_bulk * self.spacer
            _result = self._sinc(0.5 * self.length * self.mismatch) *\
                np.sin(self.reps * _phi * 0.5) / np.sin(_phi * 0.5) *\
                np.exp(1j * (self.reps - 1) * 0.5 * _phi)
        self.phasematching = _result
        return _result
