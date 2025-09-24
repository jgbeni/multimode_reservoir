#
# Copyright 2018
# Integrated Quantum Optics group, Paderborn University
# All rights reserved.
#
# author: Benjamin Brecht
# mail: benjamin.brecht@uni-paderborn.de
#

from numpy import exp, pi, sqrt, shape, zeros, log
from scipy.special import hermite,factorial
import matplotlib.pyplot as plt
import numpy as np

class Pump(object):
    """ Provides the pump function of a nonlinear process.

    === Public methods ===
    pump -- generates the pump function from the provided
            parameters

    === Private methods ===
    _hermite_mode -- normalised Hermite-Gaussian function

    === Public Variables ===
    pump_center -- pump central wavelength [m] (default: None)
    pump_wavelength -- matrix containing the pump wavelengths
                       in the signal and idler frequency plane [m]
                       (default: None)
    pump_width -- the intensity FWHM of the pump pulses [m] (default: None)
    offset -- the offset from the actual pump center [m] (default: 0)
    signal_wavelength -- matrix containing the signal wavelengths [m]
                         (default: None)
    idler_wavelength -- matrix containing the idler wavelengths [m]
                        (default: None)
    type -- keyword defining the pump type; must be in ['normal',
            'filtered', 'custom'] (default: 'normal')
    process -- nonlinear process (default: 'PDC')
    pump_delay -- temporal pump delay with respect to a reference [s]
                  (default: 0)
    pump_chirp -- quadratic chirp parameter of the pump pulse [s**2]
                  (default: 0)
    pump_temporal_mode -- temporal mode order of the pump pulse
                          (default: 0)
    pump_filter_width -- intensity FWHM of a spectral filter [m]
                         (default: 100)
    sol -- speed of light [m] (default: 299792458)

    === Private Variables ===
    _result -- about every calculation result
    _pump_function -- matrix containing the pump function
    _filter -- matrix containing the filter function
    """
    def __init__(self, pump_center=None, pump_wavelength=None,
                 pump_width=None, offset=0, signal_wavelength=None,
                 idler_wavelength=None, pump_type="normal",process="PDC", a_n=[1], delta_n=[0],
                 pump_delay=0, pump_chirp=0,
                 pump_temporal_mode=0, pump_filter_width=100):
        """ Initialise a pump with default parameters. """
        self.process = process
        self.pump_center = pump_center
        self.pump_wavelength = pump_wavelength
        self.pump_width = pump_width
        self.pump_type = pump_type
        self.offset = offset
        self.signal_wavelength = signal_wavelength
        self.idler_wavelength = idler_wavelength
        self.type = pump_type
        self.a_n = a_n
        self.delta_n = delta_n
        self.pump_delay = pump_delay
        self.pump_chirp = pump_chirp
        self.pump_temporal_mode = pump_temporal_mode
        self.pump_filter_width = pump_filter_width
        self.sol = 299792458.0
        self.pump_interv = 1

    def _hermite_mode(self, x):
        if self.offset == 0:
            """ A normalised Hermite-Gaussian function """
            _result = hermite(self.pump_temporal_mode)((self.pump_center - x) /
                                                       self.pump_width) *    \
                exp(-(self.pump_center - x)**2 / (2 * self.pump_width**2)) /\
                sqrt(factorial(self.pump_temporal_mode) * sqrt(pi) *
                     2**self.pump_temporal_mode * self.pump_width)
        else:
            _tmp = self.pump_center - self.offset
            _result = hermite(self.pump_temporal_mode)((_tmp - x) /
                                                       self.pump_width) *    \
                exp(-(_tmp - x)**2 / (2 * self.pump_width**2)) /\
                sqrt(factorial(self.pump_temporal_mode) * sqrt(pi) *
                     2**self.pump_temporal_mode * self.pump_width)
            _tmp = self.pump_center + self.offset
            _result += hermite(self.pump_temporal_mode)((_tmp - x) /
                                                        self.pump_width) *    \
                exp(-(_tmp - x)**2 / (2 * self.pump_width**2)) /\
                sqrt(factorial(self.pump_temporal_mode) * sqrt(pi) *
                     2**self.pump_temporal_mode * self.pump_width)
        return _result

    def pump(self):
        """ Calculates the pump function

        === Returns ===
        _pump_function -- matrix containing the pump function in
                          signal and idler frequecy plane
        """
        self.pump_width /= 2 * sqrt(log(2))
        if self.process.upper() in ['PDC', 'BWPDC']:
            self.pump_wavelength = 1.0 / (1.0 / self.signal_wavelength +
                                          1.0 / self.idler_wavelength)
        elif self.process.upper() == 'SFG':
            self.pump_wavelength = 1.0 / (1.0 / self.idler_wavelength -
                                          1.0 / self.signal_wavelength)
        elif self.process.upper() == 'DFG':
            self.pump_wavelength = 1.0 / (1.0 / self.signal_wavelength -
                                          1.0 / self.idler_wavelength)
        if self.type.upper() == 'NORMAL':
            _pump_function = self._hermite_mode(self.pump_wavelength) *\
                exp(1j * 2 * pi * self.sol / self.pump_wavelength *
                    self.pump_delay) *\
                exp(1j * (2 * pi * self.sol / self.pump_wavelength)**2 *
                    self.pump_chirp)
        elif self.type.upper() == 'FILTERED':
            self.pump_filter_width *= sqrt(2)
            _filter = zeros(shape(self.pump_wavelength), float)
            print(shape(self.pump_wavelength))
            for i in range(len(self.signal_wavelength)):
                print (i)
                for j in range(len(self.idler_wavelength)):
                    if self.pump_wavelength[j, i] < self.pump_center -\
                            0.5 * self.pump_filter_width:
                        pass
                    elif self.pump_wavelength[j, i] <= self.pump_center +\
                            0.5 * self.pump_filter_width:
                        _filter[j, i] = 1
                    else:
                        pass
            _pump_function = self._hermite_mode(self.pump_wavelength) *\
                exp(1j * 2 * pi * self.sol / self.pump_wavelength *
                    self.pump_delay) *\
                exp(1j * (2 * pi * self.sol / self.pump_wavelength)**2 *
                    self.pump_chirp) * _filter
        elif self.type.upper() == 'CUSTOM':
            from utils.custom_pump_frexel import custom_pump
            pwl=float(self.pump_center)
            interv=np.arange(pwl-10e-9,pwl+10e-9,0.01e-9)
#             fig = plt.figure(figsize=(3, 2))
#             plt.plot(interv, np.real(custom_pump(interv, self.pump_center,
#                                          self.pump_width, self.a_n, self.delta_n) ))
#             plt.title(f'a, delta(ps) = {[round(a, 2) for a in self.a_n], [round(d*1e12,2) for d in self.delta_n]}')
#             plt.show()
            self.pump_interv=(custom_pump(interv, self.pump_center,
                                         self.pump_width, self.a_n, self.delta_n)) 
#             print(len(self.pump_interv))
            _pump_function = custom_pump(self.pump_wavelength,
                                         self.pump_center,
                                         self.pump_width, self.a_n, self.delta_n)
            
        return _pump_function
