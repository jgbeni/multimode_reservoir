# This code takes the relevant parts from:

## Integrated Quantum Optics group, Paderborn University
## All rights reserved.
##
## author: Benjamin Brecht
## mail: benjamin.brecht@uni-paderborn.de
##

# It's used to give the decision variables for a genetic algorithm or similar
# Get all the information in the main code!
# The main calculations are started from the executeCore function and needs following input parameters.
# If the parameters are not defined, the named default-values are chosen:
#
"""
    Execute the core calculation with the given input parameters.

    Parameters:
    xn (list): xn
    length (float): Length of the crystal [m]. Default: 15E-3 m
    type (str): Type of nonlinear process ('0', 'I', 'II'). Default: '0'
    lambda_central (float): Central wavelength [m]. Default: 1560E-9 m
    pump_polarization (str): Polarization of the pump. Default: "V"
    signal_polarization (str): Polarization of the signal. Default: "V"
    idler_polarization (str): Polarization of the idler. Default: "V"
    pump_width (float): Pump width at FWHM [m]. Default: 2E-9 m
    width (float): Width of the crystal [m]. Default: 3e-6 m
    height (float): Height of the crystal [m]. Default: 3e-6 m
    crystal_type (str): Type of nonlinear crystal. Default: 'KTP'
    number_relevant_modes (int): Number of relevant modes for the return value. Default: 5

    Returns:
    relevant_overlaps (list): List of relevant overlaps for upper given modes.
    relevant_schmidt_numbers (list): List of relevant Schmidt numbers for upper given modes.
    K (int): The value of K.
"""


import numpy as np
import matplotlib.pyplot as plt
from utils.Process import ParametricProcess
from scipy.signal import find_peaks


def executeCore(length=1e-3, type_='0', pm_type = "sinc", lambda_central=795E-9,
                pump_polarization="V", window=300e-9, signal_polarization="V", idler_polarization="V", steps=700,
                pump_width=0.5E-9,pump_type = 'normal', process='PDC',
        a_n=[1], delta_n=[0], width=3e-6, height=3e-6, crystal_type='KTP'):
    
    grating = np.inf
    Type = type_
    pm_type = "sinc"
    process = 'PDC'
    pump_center = lambda_central / 2  # pump central wavelength in [m]
    pump_temporal_mode = 0  # temporal mode order of the pump
    signal_center = lambda_central  # signal central wavelength in [m]
    signal_start = lambda_central - window  # start of signal plot range in [m]
    signal_stop = lambda_central + window  # end of signal plot range in [m]
    idler_start = lambda_central - window  # start of idler plot range in [m]
    idler_stop = lambda_central + window  # end of idler plot range in [m]
    signal_steps = steps  # points along the signal axis
    idler_steps = steps  # points along the idler axis
    rebin_factor = 1

    temp = 40
    save_modes = False
    save_schdmidt_coeff = True

    # ------------------------------
    # Set up the actual calculation.
    # ------------------------------
    pp = ParametricProcess()  # Create a class instance
    pp.set_parameters(
        length=length,
        temperature=temp,
        grating=grating,
        wg_width=width,  # metallic waveguide with in [m]
        wg_height=height,  # metallic waveguide height in [m]
        pump_polarization=pump_polarization,
        signal_polarization=signal_polarization,
        idler_polarization=idler_polarization,
        pm_type=pm_type,
        process=process,
        pump_center=pump_center,
        pump_width=pump_width,
        pump_temporal_mode=pump_temporal_mode,
        pump_type = pump_type,
        a_n=a_n, delta_n=delta_n,
        signal_center=signal_center,
        signal_start=signal_start,
        signal_stop=signal_stop,
        idler_start=idler_start,
        idler_stop=idler_stop,
        signal_steps=signal_steps,
        idler_steps=idler_steps,
        crystal=crystal_type)
    # Set the defined calculation parameters.
    pp.calculate('svd')  # calculate the JSA and decomposition.
    pp.rebin(rebin_factor)  # rebin the JSA for faster plotting.
    results = pp.results

    return results