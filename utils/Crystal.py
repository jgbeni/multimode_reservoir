#
# Copyright 2018
# Integrated Quantum Optics group, Paderborn University
# All rights reserved.
#
# author: Benjamin Brecht
# mail: benjamin.brecht@uni-paderborn.de
#

from numpy import sqrt, loadtxt, asarray, cos, sin


class Crystal(object):
    """ Provides Sellmeier equations of nonlinear waveguides.

    === Public Methods ===
    generate_sellmeier -- generates effective refractive indices of
                          waveguides (metallic model) or refractive
                          indices retrieved from user-provided Sell-
                          meier equations
    nh -- refractive index for horizontal polarization as a function
          of wavelength; for 'LN' and 'KTP', the spatial mode orders
          have to be provided as well.
    nv -- refractive index for vertical polarization as a function
          of wavelength; for 'LN' and 'KTP', the spatial mode orders
          have to be provided as well.

    === Private Methods ===
    _crosscheck_process -- for 'LN' and 'KTP', this method automatically
                           checks whether the desired process is allowed;
                           uses crystal cut, propagation and polarizations

    === Public Variables ===
    crystal -- nonlinear material (default: 'LN')
    crystal_cut -- cut axis of the crystal (default: 'z')
    propagation -- optical axis in crystal coordinates (default: 'x')
    temperature -- sample temperature in [deg C] (default: 25)
    wg_width -- width of the waveguide in [m] (default: 6E-6)
    wg_height -- height of the waveguide in [m] (default: 4E-6)
    pump_polarization -- polarization of the pump (default: 'v')
    signal_polarization -- polarization of the signal (default: 'v')
    idler_polarization -- polarization of the idler (default: 'v')
    hfile -- file with user-provided indices for h-polarization (default: None)
    vfile -- file with user-provided indices for v-polarization (default: None)

    """
    def __init__(self, crystal='LN', temperature=25, hfile=None, vfile=None,
                 crystal_cut="Z", propagation="X", pump_polarization="V",
                 signal_polarization="V", idler_polarization="V",
                 wg_width=4E-6, wg_height=6E-6):
        """ Initialise a crystal with default values. """
        self.crystal = crystal
        self.temperature = temperature
        self.hfile = hfile
        self.vfile = vfile
        self.crystal_cut = crystal_cut
        self.propagation = propagation
        self.pump_polarization = pump_polarization
        self.signal_polarization = signal_polarization
        self.idler_polarization = idler_polarization
        self.wg_height = wg_height
        self.wg_width = wg_width

        self.error_code = 0

    def generate_sellmeier(self):
        """ Calculate the Sellmeier equations for h- and v-polarized light.

        === Returns ===
        (nh, nv) -- tuple containing the Sellmeier equations
        """
        if self.crystal in ['uniaxial', 'uniaxial_temp']:
            # If the crystal uses user-provided indices, try to load the
            # respective files
            try:
                _ch = loadtxt(self.hfile)
                _cv = loadtxt(self.vfile)
            except:
                _message = "Could not load custom Sellmeier coefficients."
                raise ValueError(_message, [self.hfile, self.vfile])
            if self.crystal == 'uniaxial':
                # This is not temperature dependent
                self.nh = lambda wl: sqrt(_ch[0] + _ch[1] / ((wl * 1E6)**2 -
                                          _ch[2]) + _ch[3] / ((wl * 1E6)**2 -
                                          _ch[4]))

                self.nv = lambda wl: sqrt(_cv[0] + _cv[1] / ((wl * 1E6)**2 -
                                          _cv[2]) + _cv[3] / ((wl * 1E6)**2 -
                                          _cv[4]))

            elif self.crystal == 'uniaxial_temp':
                # This is temperature dependent; the Sellmeier coefficients
                # are themselves functions of temperature
                _H1 = _ch[0][0] + _ch[0][1] * self.temperature +\
                      _ch[0][2] * self.temperature**2
                _H2 = _ch[1][0] + _ch[1][1] * self.temperature +\
                    _ch[1][2] * self.temperature**2
                _H3 = _ch[2][0] + _ch[2][1] * self.temperature +\
                    _ch[2][2] * self.temperature**2
                _H4 = _ch[3][0] + _ch[3][1] * self.temperature +\
                    _ch[3][2] * self.temperature**2
                _H5 = _ch[4][0] + _ch[4][1] * self.temperature +\
                    _ch[4][2] * self.temperature**2
                _V1 = _cv[0][0] + _cv[0][1] * self.temperature +\
                    _cv[0][2] * self.temperature**2
                _V2 = _cv[1][0] + _cv[1][1] * self.temperature +\
                    _cv[1][2] * self.temperature**2
                _V3 = _cv[2][0] + _cv[2][1] * self.temperature +\
                    _cv[2][2] * self.temperature**2
                _V4 = _cv[3][0] + _cv[3][1] * self.temperature +\
                    _cv[3][2] * self.temperature**2
                _V5 = _cv[4][0] + _cv[4][1] * self.temperature +\
                    _cv[4][2] * self.temperature**2

                self.nh = lambda wl: sqrt(_H1 + _H2 / ((wl * 1E6)**2 - _H3) +
                                          _H4 / ((wl * 1E6)**2 - _H5))
                self.nv = lambda wl: sqrt(_V1 + _V2 / ((wl * 1E6)**2 - _V3) +
                                          _V4 / ((wl * 1E6)**2 - _V5))

        elif self.crystal.upper() in ['KTP', 'PPKTP', 'LN', 'PPLN']:
            # If the crystal is lithium niobate or potassium
            # titanyl phosphate, check whether the desired process
            # is allowed
            self._crosscheck_process()
            # Determine the crystal axes for h- and v-polarization
            _v_index = self.crystal_cut.upper()
            _h_index = ["X", "Y", "Z"]
            _h_index.remove(self.crystal_cut.upper())
            try:
                _h_index.remove(self.propagation.upper())
            except:
                _msg = "Crystal cut and propagation must be different."
                raise ValueError(_msg, self.crystal_cut, self.propagation)
            _h_index = _h_index[0]
            # Create a dictionary that translates crystal axes to integers;
            # this is the pythonic way of implementing a case structure
            _indices = {"X": 0, "Y": 1, "Z": 2}
            # Sellmeier equations for lithium niobate; taken from Edwards &
            # Lawrence and from Jundt
            if self.crystal.upper() in ['LN', 'PPLN']:
                _O1 = 4.9048
                _O2 = 0.11775
                _O3 = 2.2314e-8
                _O4 = 0.21802
                _O5 = 2.9671e-8
                _O6 = 0.027153
                _O7 = 2.1429e-8
                _E1 = 5.35583
                _E2 = 4.629e-7
                _E3 = 0.100473
                _E4 = 3.862e-8
                _E5 = 0.20692
                _E6 = 0.89e-8
                _E7 = 100
                _E8 = 2.657e-5
                _E9 = 11.34927
                _E10 = 1.5334e-2
                _TO = (self.temperature - 24.5) * (self.temperature + 570.50)
                _TE = (self.temperature - 24.5) * (self.temperature + 570.82)
                # Lithium niobate is a uniaxial crystal and thus has only
                # two different refractive indices, namely ordinary and
                # extra-ordinary
                self.no = lambda wl, mode_width, mode_height: \
                    sqrt(_O1 + (_O2 + _O3 * _TO) / ((wl * 1E6)**2 - (_O4 -
                         _O5 * _TO)**2) - _O6 * (wl * 1E6)**2 + _O7 * _TO -
                         (wl * (mode_height + 1) / (2 * self.wg_height))**2 -
                         (wl * (mode_width + 1) / (2 * self.wg_width))**2)
                self.ne = lambda wl, mode_width, mode_height: \
                    sqrt((_E1 + _E2 * _TE + (_E3 + _E4 * _TE) /
                         ((wl * 1e6)**2 - (_E5 - _E6 * _TE)**2) + (_E7 + _E8 *
                         _TE) / ((wl * 1e6)**2 - _E9**2) - _E10 *
                         (wl * 1e6)**2) - (wl * (mode_height + 1) /
                         (2 * self.wg_height))**2 -
                         (wl * (mode_width + 1) / (2 * self.wg_width))**2)
                # Associate the h- and v-polarized indices with ordinary
                # and extra-ordinary according to the encoding from above
                self.nh = [self.no, self.no, self.ne][_indices[_h_index]]
                self.nv = [self.no, self.no, self.ne][_indices[_v_index]]
            # Sellmeier equations for potassium titanyl phosphate; taken from
            # Takaoka ...
            elif self.crystal.upper() in ['KTP', 'PPKTP']:
                _X1 = 3.29100
                _X2 = 0.04140
                _X3 = 0.03978
                _X4 = 9.35522
                _X5 = 31.45571
                _Y1 = 3.45018
                _Y2 = 0.04341
                _Y3 = 0.04597
                _Y4 = 16.98825
                _Y5 = 39.43799
                _Z1 = 4.59423
                _Z2 = 0.06206
                _Z3 = 0.04763
                _Z4 = 110.80672
                _Z5 = 86.12171
                # Potassium titanyl phosphate is a biaxial crystal and thus
                # has three different refractive indices, typically referred
                # to as 'x', 'y', and 'z'
                self.nx = lambda wl, mode_width, mode_height: \
                    sqrt(_X1 + _X2 / ((wl * 1E6)**2 - _X3) +
                         _X4 / ((wl * 1E6)**2 - _X5) -
                         (wl * (mode_height + 1.0) / (2 * self.wg_height))**2 -
                         (wl * (mode_width + 1.0) / (2 * self.wg_width))**2)
                self.ny = lambda wl, mode_width, mode_height: \
                    sqrt(_Y1 + _Y2 / ((wl * 1E6)**2 - _Y3) +
                         _Y4 / ((wl * 1E6)**2 - _Y5) -
                         (wl * (mode_height + 1.0) / (2 * self.wg_height))**2 -
                         (wl * (mode_width + 1.0) / (2 * self.wg_width))**2)
                self.nz = lambda wl, mode_width, mode_height: \
                    sqrt(_Z1 + _Z2 / ((wl * 1E6)**2 - _Z3) +
                         _Z4 / ((wl * 1E6)**2 - _Z5) -
                         (wl * (mode_height + 1.0) / (2 * self.wg_height))**2 -
                         (wl * (mode_width + 1.0) / (2 * self.wg_width))**2)
                # Associate the h- and v-polarized indices with 'x', 'y',
                # and 'z' according to the encoding from above.
                self.nh = [self.nx, self.ny, self.nz][_indices[_h_index]]
                self.nv = [self.nx, self.ny, self.nz][_indices[_v_index]]
        elif self.crystal.upper() == 'BBO':
            _O1 = 2.7366122
            _O2 = 0.0185720
            _O3 = 0.0178746
            _O4 = 0.0143756
            _E1 = 2.3698703
            _E2 = 0.0128445
            _E3 = 0.0153064
            _E4 = 0.0029129
            self.no = lambda wl: \
                sqrt(_O1 + _O2 / ((wl * 1E6)**2 - _O3) - _O4 * (wl * 1E6)**2)
            self.ne = lambda wl: \
                sqrt(_E1 + _E2 / ((wl * 1E6)**2 - _E3) - _E4 * (wl * 1E6)**2)
            self.nh = lambda wl, theta: self.no(wl)
            self.nv = lambda wl, theta: \
                sqrt(1.0 / (cos(theta)**2 / self.no(wl)**2 +
                     sin(theta)**2 / self.ne(wl)**2))
        else:
            _msg = (
                "Crystal must be in (KTP, LN, BBO, ",
                "uniaxial, uniaxial_temp).")
            raise ValueError(_msg, self.crystal)
        return (self.nh, self.nv)

    def _crosscheck_process(self):
        """ Check whether the desired process is allowed. """
        # Determine the crystal axes for h- and v-polarization
        _h_index = ["X", "Y", "Z"]
        _h_index.remove(self.crystal_cut.upper())
        _h_index.remove(self.propagation.upper())
        _h_index = _h_index[0]
        _v_index = self.crystal_cut.upper()
        # Dictionary that translates polarizations into crystal axes
        _polarizations = {"H": _h_index, "V": _v_index}
        # Dictionary that translates crystal axes and their combinations
        # to contracted tensor element identifiers according to the
        # Kleinmann (?) symmetry
        _tensor_elements = {"X": 0, "XX": 0,
                            "Y": 1, "YY": 1,
                            "Z": 2, "ZZ": 2,
                            "XY": 5, "YX": 5,
                            "YZ": 3, "ZY": 3,
                            "XZ": 4, "ZX": 4}
        # Generate the polarization identifier for the photons
        _photons = _polarizations[self.signal_polarization.upper()] +\
            _polarizations[self.idler_polarization.upper()]
        # Implement the nonlinear tensors in a symbolic notation
        if self.crystal.upper() in ['LN', 'PPLN']:
         #   _tensor = asarray([[0, 0, 0, 1, 0, 1],
          #                     [1, 1, 0, 0, 1, 0],
          #                     [1, 1, 1, 0, 0, 0]])
            _tensor = asarray([[0, 0, 0, 0, 1, 1],
                               [1, 1, 0, 1, 0, 0],
                               [1, 1, 1, 0, 0, 0]])
         
        elif self.crystal.upper() in ['KTP', 'PPKTP']:
         #   _tensor = asarray([[0, 0, 0, 0, 0, 1],
          #                     [0, 0, 0, 0, 1, 0],
          #                     [1, 1, 1, 0, 0, 0]])
            _tensor = asarray([[0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 1, 0, 0],
                               [1, 1, 1, 0, 0, 0]])
         
        # Retrieve the tensor element identifiers for the pump and photons
        _pump = _tensor_elements[
            _polarizations[self.pump_polarization.upper()]]
        _photons = _tensor_elements[_photons]
        # Check whether there is a tensor element for the combination of pump
        # and photons and print a warning if not
        if _tensor[_pump, _photons] == 0:
            _msg = "Warning! The process you are calculating "
            _msg += "is not allowed in this material."
            self.error_code = 1
            print(_msg)
