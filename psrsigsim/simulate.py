"""simulate.py, a wrapper for all other modules in simulation.

For init method docstrings, under parameter section, the information
beneath format (Key : Value type/possible values) pertains not to
function parameters, but to the required contents of the dictionary parameter.

Units for V_ISS etc.?
"""
import psrsigsim as PSS
import psrsigsim as PSS
from psrsigsim import Telescope
import sys
import pint
# from pint.models import polycos
from pint import models
from pint import utils
import os.path

d_path = os.path.dirname(__file__)

default_path = os.path.join(d_path, './data/')
class Simulation(object):
    """This class simulates an observing run given a pulsar name or a
    dictionary of parameters.

    Once the simulation is called, the different classes
    (signal, pulsar,ism,scint,telescope) can be initialized with the
    initialization functions. init_signal and init_pulsar are the only required
    functions to run the simulation. If a pulsar name is input, then these
    function are ran automatically.
    The required parameters needed for the simulation are:

    f0 : float
        Central frequency (MHz).
    F0 : float
        Pulsar Spin Frequency (Hz).
    bw : float
        Bandwidth (MHz).
    Nf :int
        Number of frequency bins.
    ObsTime : float
        Total observation time in milliseconds.
    f_samp : float
        Sampling frequency (MHz).
    SignalType : {'intensity', 'voltage'}
        'intensity' which carries a Nf x Nt filterbank of pulses or 'voltage'
         which carries a 4 x Nt array of voltage vs. time pulses representing
         4 stokes channels.
    dm : float
        Dispersion measure.
    V_ISS : float
        Interstellar scintillation velocity.
    scint_bw : float
        Scintillation bandwidth.
    scint_timescale : float
        Scintillation timescale.
    pulsar : str
        Pulsar name
    telescope : {'GBT', 'Arecibo'}
        Telescope name (GBT or Arecibo).
    freq_band : {327 ,430, 820, 1400, 2300}
        Frequency band.
    aperture : float
        Telescope aperture (m).
    area : float
        Collecting area (m^2).
    Tsys : float
        System temp (K), total of receiver, sky, spillover, etc. (only needed for noise).
    name : {'GBT', 'Arecibo'}
        GBT or Arecibo.
    tau_scatter : float
        Scattering time (ms).
    radiometer_noise : bool
        Changes whether radiometer noise is included or not. True/False.
    to_DM_broaden : bool
        Changes whether dm broadening is included. True/False.

    Parameters retrieved when pulsar name is input : F0, dm, scint_bw, scint_timescale.

    Parameters
    ----------
    psr : str, optional
        Pulsar name.
    sim_telescope : {'GBT', 'Arecibo'}, optional
        Specified telescope class is initialized to "observe" simulated pulsar.
    sim_ism : bool, optional
        If True, initializes ism class, adding dispersion effects.
    sim_scint : bool, optional
        If True, scintillate class is initialized to scintillate pulsar signal.
    sim_dict : dict, optional
        Dictionary containing parameters of the simulation. If set to default None,
        dictionaries containing parameters must be created manually and inputted
        to each init methods used by user.
    sim_file_path : str, optional
        Path to data files.

    Examples
    --------
    Simulations using pulsar name:

    >>> dictionary = {'f0':1400,'bw':400,'Nf':100,'f_samp':4,ObsTime':10,
    ...              'data_type':'int8','SignalType':"intensity",'flux':3,
    ...              'tau_scatter':6e-05,'radiometer_noise: True, to_DM_broaden: True}
    >>> s1 = Simuation(psr = 'J1614-2230',sim_telescope ='GBT', sim_ism = True,
    ...               sim_scint = True, sim_dict = None)
    >>> s1.simulate()

    Simulations using own dictionary:

    >>> dictionary = {'f0':1400,'dm':15.921200, 'F0': 218, 'bw':400,'Nf':100,
    ...               'f_samp':4,ObsTime':10,'data_type':'int8',
    ...               'SignalType':"intensity",'flux':3,'tau_scatter':6e-05,
    ...               'radiometer_noise: True, to_DM_broaden: True}
    >>> s2 = Simulation(psr = None,sim_telescope = 'Arecibo',sim_ism = None,
    ...                 sim_scint = None,sim_dict = dictionary)
    >>> s2.init_signal()
    >>> s2.init_pulsar()
    >>> s2.init_ism()
    >>> s2.init_telescope()
    >>> s2.simulate()
    """
    def __init__(self, psr = None, sim_telescope = None, sim_ism = False, sim_scint = False, \
                 sim_dict = None, sim_file_path = default_path ):

        self.psr = psr
        self.sim_telescope = sim_telescope
        self.sim_ism = sim_ism
        self.sim_scint = sim_scint
        self.sim_dict = sim_dict
        self.sim_file_path = sim_file_path



        if isinstance(psr, str):
            self.param_dict = self._set_pulsar_dict()
            print('Five warnings should appear. Warnings that are normal are: Unrecognized parfile line "SOLARN0", "T2CMETHOD", "MODE", "INFO -f". Warning about "DDK model" is also normal.  ')
            if 'F0' in self.param_dict.keys() and 'F0' in self.sim_dict.keys():
                print( "Two 'F0' values input. Simulator will default to input dictionary value.")
            if 'dm' in self.param_dict.keys() and 'dm' in self.sim_dict.keys():
                print( "Two 'dm' values input. Simulator will default to input dictionary value.")

            self.param_dict.update(self.sim_dict)
            self.sim_dict = self.param_dict


            self.init_signal()
            self.init_pulsar()
            if sim_ism:
                self.init_ism()

            if sim_scint:
                self.init_scint()

            if sim_telescope:
                self.init_telescope()







    def init_signal(self,signal_dict = None):
        """This is a function that initializes a signal class using a dictionary
         of parameters.

        It either uses the dictionary set when Simulation() is called, if
        signal_dict is left as None, or a dictionary that is input directly
        into the init_signal function. mode is automatically set to 'simulate'.

        Parameters
        ----------
        signal_dict : dict, optional
            Required keys of this dictionary are parameters
            (Key : Value type/possible values)

            f0 : float
                Central frequency (MHz).
            bw : float
                Bandwidth (MHz).
            Nf : int
                Number of frequency bins.
            f_samp : float
                Sampling frequency (MHz).
            ObsTime : float
                Total observation time in milliseconds.
            data_type : {'int8', 'int16'}
                Automatically changed to 'uint8' or 'uint16' if intensity signal.
            SignalType : {'voltage', 'intensity'}
                'intensity' which carries a Nf x Nt filterbank of pulses or 'voltage'
                which carries a 4 x Nt array of voltage vs. time pulses representing
                4 stokes channels
        """
        d = self._check_simul_dict(signal_dict)
        self.signal = PSS.Signal(f0 = d['f0'],bw = d['bw'], Nf = d['Nf'], f_samp = d['f_samp'] , ObsTime = d['ObsTime']\
                                    ,data_type = d['data_type'],SignalType = d['SignalType'], mode = 'simulate')


    def init_pulsar(self,pulsar_dict = None):
        """This is a function that initializes the pulsar class using a dictionary
        of parameters.

        It either uses the dictionary set when Simulation() is called,
        if pulsar_dict is left as None, or a dictionary that is input directly
        into the init_pulsar function.

        Parameters
        ----------
        pulsar_dict : dict, optional
            Required keys of this dictionary are parameters
            (Key : Value type/possible values)

            F0 : float
                Pulsar spin frequency (Hz).
            flux : float
                Mean flux density of pulsar (mJy).
        """
        d =  self._check_simul_dict(pulsar_dict)
        self.pulsar = PSS.Pulsar(self.signal, period = 1000/float(d['F0']) , flux = d['flux']) # in units of milliseconds


    def init_ism(self,ism_dict = None):
        """This is a function that initializes the ism class using a dictionary
        of parameters.

        It either uses the dictionary set when Simulation() is called, if ism_dict
        is left as None, or a dictionary that is input directly into the
        init_ism function.

        Parameters
        ----------
        ism_dict : dict, optional
            Required key of this dictionary is parameter
            (Key : Value type/possible values)

            dm : float
            Dispersion measure.
        """
        d = self._check_simul_dict(ism_dict)
        self.ISM = PSS.ISM(self.signal, DM = d['dm'])





    def init_scint(self,scint_dict = None):
        """This is a function that initializes the scintillate class using a dictionary
        of parameters.

        It either uses the dictionary set when Simulation() is called,
        if scint_dict is left as None, or a dictionary that is input directly
        into the init_scint function.

        Parameters
        ----------
        scint_dict : dict, optional
            Required keys of this dictionary are parameters
            (Key : Value type/possible values)

            V_ISS : float
                Interstellar scintillation velocity.
            scint_bw : float.
                Scintillation bandwidth
            scint_timescale : float
                Scintillation timescale.
            pulsar : str
                Pulsar name.
            to_use_NG_pulsar : bool
                True to use NG pulsar. False otherwise.
            telescope : {'GBT', 'Arecibo'}
                Telescope name. Either Green Bank telescope or Arecibo telescope.
            freq_band : float
                Frequency band.
        """
        d = self._check_simul_dict(scint_dict)

        try:
            self.Scint = PSS.scintillate(self.signal, V_ISS=d['V_ISS'], scint_bw=d['scint_bw'], scint_timescale=d['scint_timescale'],\
                pulsar=None, to_use_NG_pulsar=False, telescope=None, freq_band=None)

        except KeyError:
            #self.Scint = PSS.scintillate(self.signal,  V_ISS=None, scint_bw=None, scint_timescale=None,\
                        #pulsar=d['pulsar'], to_use_NG_pulsar=True, telescope=d['telescope'], freq_band=d['freq_band'])

            self.Scint = PSS.scintillate(self.signal,  V_ISS=None, scint_bw=None, scint_timescale=None,\
                        pulsar=self.psr, to_use_NG_pulsar=True, telescope=self.sim_telescope, freq_band=d['freq_band'])





    def init_telescope(self,telescope_dict = None):
        """This is a function that initializes the telescope class using a dictionary
        of parameters.

        It either uses the dictionary set when Simulation() is called, if
        telescope_dict is left as None, or a dictionary that is input directly
        into the init_telescope function. If 'GBT' or 'Arecibo' is input to
        sim_telescope then it will be automatically put
        into the simulation.

        Parameters
        ----------
        telescope_dict : dict, optional
            Required keys of this dictionary are parameters
            (Key : Value type/possible values)

            aperture : float
                Telescope aperture (m)
            area : float
                Collecting area (m^2)
            Tsys : float
                System temperature (K), total of receiver, sky, spillover, etc.
                (only needed for noise)
            name : {'GBT', 'Arecibo'}
                Telescope name.
        """
        if self.sim_telescope == 'GBT':
            self.telescope = PSS.GBT()

        elif self.sim_telescope == 'Arecibo':
            self.telescope = PSS.Arecibo()

        elif self.sim_telescope == None:

            d = self._check_simul_dict(telescope_dict)
            self.telescope = PSS.Telescope(d['aperture'],area = d['area'], Tsys = d['Tsys'], name = d['telescope'])







    def simulate(self):
        """This is the function that runs the simulation once the desired classes
        are initialized.
        """
        if hasattr(self,'ISM'):
            if 'tau_scatter' in self.sim_dict.keys():
                self.ISM.to_Scatter_Broaden_exp = True
                self.ISM.tau_scatter = self.sim_dict['tau_scatter']

            if 'DM' and 'to_DM_Broaden' in self.sim_dict.keys():
                self.ISM.to_DM_Broaden = self.sim_dict['to_DM_Broaden']




            self.ISM.finalize_ism()

            if self.signal.MetaData.to_DM_Broaden:
                tophats = PSS.ism.make_dm_broaden_tophat(self.pulsar,self.signal)
                PSS.ism.convolve_with_profile(self.pulsar,tophats)

            if self.signal.MetaData.to_Scatter_Broaden_exp:
                exponentials = PSS.ism.make_scatter_broaden_exp(self.pulsar,self.signal,self.sim_dict['tau_scatter'])
                PSS.ism.convolve_with_profile(self.pulsar,exponentials)


        self.pulsar.make_pulses(subintlen = self.signal.subintlen)


        try:
            self.ISM.disperse()

        except AttributeError:
            pass

#         try:
#             self.Scintillate = self.Scint.() # Add command once it exists
#         except AttributeError:
#             pass

        if hasattr(self,'telescope'): # if sim_telescope is not true it skips
            if self.sim_dict['radiometer_noise']:
                self.obs_signal = self.telescope.radiometer_noise(self.signal,[self.signal.Nf, self.signal.Nt],self.signal.TimeBinSize)








####Convenience Methods

    def _check_simul_dict(self,d):
        """Checks for input dictionary in init method and sets it as a dictionary
        to be used in the initialization of the class.

        This function is used to make the init methods easier to write.

        Parameters
        ----------
        d
            The input to dictionary keyword in init methods. If empty, d is assigned
            the dictionary used when Simulation() is called.

        Returns
        -------
        d : dict
            Dictionary of simulation parameters to be used in class initialization.
        str
            ValueError message if there is no parameter dictionary in both init methods
            and Simulation class constructor.
        str
            TypeError message if input dictionary is not a dictionary.
        """
        if not d:
            if self.sim_dict:
                d = self.sim_dict
            else:
                raise ValueError('Parameter dictionary needed.')
        elif isinstance(d,dict):
            pass
        else:
            raise TypeError('Input not a dictionary')


        return d


    def _set_pulsar_dict(self):
        """Takes Pulsar name, retrieves the following parameters:

        F0 : float
            Pulsar Spin Frequency (Hz)
        dm : float
            Dispersion measure
        scint_bw : float
            scintillation Bandwidth
        scint_timescale : float
            Scintillation timescale

        and returns them in a dictionary.

        Returns
        -------
        param_dict : dict
            Dictionary containing parameters to be added to sim_dict in Simulation().
        """

        p = PSS.PSS_utils.get_pint_models(self.psr, self.sim_file_path)
        param_dict = {'F0': p.F0.value, 'dm':p.DM.value}
        [scint_bw, scint_timescale] = PSS.ism.NG_scint_param(self.psr, self.sim_telescope , self.sim_dict['freq_band'])
        param_dict['scint_bw'] = scint_bw
        param_dict['scint_timescale'] = scint_timescale


        return param_dict



    def add_scint(self):
        """
        This is old code and will be superseded by adding the scintillations into
        the DAT_SCL arrays of the SUB_INT HDU in the SEARCH mode PSRFITS file.
        We keep it here until we have fully tested how that functionality works
        with folding the PSRFITS file.
        """
        scint_time = self.MD.scint_time/self.MD.scint_time_sample_rate
        scint_samples_per_obs = np.floor(self.MD.ObsTime//(scint_time*1e3))
        #print('scint_samples_per_obs',scint_samples_per_obs)
        #gain_norm = self.scint_class.gain.max() #Could set to avoid clipping, but not sure it's needed.
        gain = self.scint_class.gain# / gain_norm
        scint_end_bin = scint_samples_per_obs * scint_time*1e3 #integer number of bins in scint
        #print('scint_end_bin',scint_end_bin)
        self.start_times = np.linspace(0, scint_end_bin, scint_samples_per_obs)
        orig_profile = np.copy(self.P.profile) #* P.gamma_draw_norm
        scint_bins = int(scint_time//self.S.TimeBinSize)
        tweak = 12
        if len(self.start_times) > len(gain[0,:]):
            raise ValueError('Scattering Screen is not long enough to scintillate at this Dispersion timescale.')
        for ii, bin_time in enumerate(self.start_times) :
            self.P.profile = gain[:, ii, np.newaxis] * orig_profile * self.P.gamma_draw_norm
            #self.P.profile /= (self.P.profile.max()/tweak)
            self.P.make_pulses(bin_time, bin_time + scint_time)
            #bin = int(bin_time //self.S.TimeBinSize)
            #self.S.signal[:,bin : bin + scint_bins] = gain[:,ii,np.newaxis]*self.S.signal[:,bin: bin + scint_bins]
        #self.frig = gain[:,ii,np.newaxis] * orig_profile
