"""signal.py, a starting point for the pulsar class."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import h5py, os
# from astropy import units as u
from . import PSS_plot


class MetaData(object):
    """The MetaData class to contain information about the signal.

    Information is continuously added to metadata via dictionaries
    from other modules.

    Attributes
    ----------
    f0 : float
        Central frequency of bandwidth (MHz).
    bw : float
        Bandwidth (MHz).
    Nf : int
        Number of frequency bins.
    Nt : int
        Number of time/phase bins.
    ObsTime : float
        Total observation time in milliseconds.
    """
    def __init__(self):
        self.f0 = None  # central freq (MHz)
        self.bw = None  # bandwidth (MHz)
        self.Nf = None  # number of frequency bins
        self.Nt = None  # number of time/phase bins
        self.ObsTime = None  # Total time in milliseconds

    def AddInfo(self, Info):
        """Function to add a dictionary of information into the metadata.

        Parameters
        ----------
        Info : dict
            A dictionary containing signal parameters to be added to metadata.
        """
        for ii, jj in Info.items():
            setattr(self, ii, jj)

        #TODO Add error message if someone trys to overlap parameters


class Signal(object):
    """The Signal class to contain metadata of parameters and signal data.

    Additional parameter information about signal is added to metadata,
    some as properties, and either a NumPy 2d-array or HDF5 file is created to
    store signal data.

    Parameters
    ----------
    mode : {'explore', 'simulate'}, optional
        'explore' mode to add effects individually.
        'simulate' to add selected effects at once.

    Attributes
    ----------
    f0
    bw
    Nf
    Nt
    ObsTime
    data_type
    SignalType
    f_samp : float
        Sampling frequency (MHz)
    MetaData : instance
        Instance of metadata class inherited to store signal parameters.
    SignalDict : dict
        Dictionary of signal parameters which is added to metadata.
    Npols : int
        Number of polarizations.
    TimeBinSize : float
        Length of each time/phase bin in milliseconds.
    freqBinSize : float
        Length of each frequency bin (MHz).
    first_freq : float
        First/lowest bandwidth frequency (MHz).
    last_freq : float
        Last/highest bandwidth frequency (MHz).
    freq_Array : array_like
        Array of frequencies of length Nf within bandwidth, excluding frequencies
        at both extremes of bandwidth.
    signal : array_like
        2d numpy array to store signal data.
        If data size exceeds 2.048GB, data is stored as HDF5 file.
    """
    def __init__(self, f0=1400, bw=400, Nf=20, f_samp=1, ObsTime=200,
                 data_type='float32', SignalType="intensity",
                 mode='explore', subintlen = False):

        """initialize Signal(), executed at assignment of new instance
        data_type = 'int8' or 'int16' supported.
                    Automatically changed to 'uint8' or 'uint16' if intensity
                    signal.
        @param f0 -- central frequency (MHz)
        @param bw -- bandwidth (MHz)
        @param f_samp -- sampling frequency (MHz)
        @param Nf -- number of freq. bins
        @param Nt -- number of phase bins
        Totime = total time of the observation in milliseconds
        SignalType = 'intensity' which carries a Nf x Nt filterbank of pulses
                     or 'voltage' which carries a 2 x Nt array of voltage vs.
                     time pulses representing 4 stokes channels
        BRENT HACK: added subint parameter
        """

        self.MetaData = MetaData()
        self.f0 = f0  # (MHz)
        self.bw = bw  # (MHz)
        self.f_samp = f_samp  # (MHz)
        self.data_type = data_type
        self.SignalType = SignalType
        self.SignalDict = {}
        self.ObsTime = ObsTime   # Total time in milliseconds
        self.subintlen = subintlen # time in seconds
        # BRENT HACK: Change number of timebins for fold mode pulses
        if subintlen:
            # Edit sampling rate if subints, assume 2048 bins per subint for now
            self.f_samp = 2048.0/self.subintlen
            # Basically samples per subint now?
            Nt = int((self.ObsTime*1e-3/self.subintlen)*2048.0)+1
        else:
            Nt = int(self.ObsTime*1e-3 * self.f_samp*1e6)+1

        if Nt % 2 == 0:  # Make signal even in length (for FFTs)
            self.Nt = Nt  # phase bins
        else:
            self.Nt = Nt + 1

        if SignalType == 'voltage':
            self.Nf = int(Nf)
            self.Npols = 2  # Easy way to make 2 channels of voltage
            rows = self.Npols

            if data_type == 'float32':
                self.data_type = 'float32'
                self.SignalDict['gauss_draw_max'] = 200
                self.SignalDict['gamma_draw_max'] = 200
                self.SignalDict['data_type'] = np.float32

            elif data_type == 'int8':
                self.data_type = 'int8'
                self.SignalDict['gauss_draw_max'] = np.iinfo(np.int8).max
                self.SignalDict['data_type'] = np.int8
                # Set the correct standard deviation for the
                # pulse draws depending on data type.

            elif data_type == 'int16':
                self.data_type = 'int16'
                self.SignalDict['gauss_draw_max'] = np.iinfo(np.int16).max
                self.SignalDict['data_type'] = np.int16

        elif SignalType == 'intensity':
            self.Nf = int(Nf)
            self.Npols = 1
            rows = self.Nf

            if data_type == 'float32':
                self.data_type = 'float32'
                self.SignalDict['gauss_draw_max'] = 200
                self.SignalDict['gamma_draw_max'] = 200
                self.SignalDict['data_type'] = np.float32

            elif data_type == 'int8':
                self.data_type = 'uint8'
                self.SignalDict['gamma_draw_max'] = np.iinfo(np.uint8).max
                self.SignalDict['data_type'] = np.uint8

            elif data_type == 'int16':
                self.data_type = 'uint16'
                self.SignalDict['gamma_draw_max'] = np.iinfo(np.uint16).max
                self.SignalDict['data_type'] = np.uint16

        else:
            err_msg = 'SiganlType: {0} + data_type: {1} not supported'
            raise ValueError(err_msg.format(SignalType, data_type))

        self.TimeBinSize = self.ObsTime/self.Nt
        self.freqBinSize = self.bw/self.Nf
        self.first_freq = self.f0 - self.freqBinSize * self.Nf/2
        if self.first_freq == 0.0:
            self.first_freq = self.first_freq + self.freqBinSize * 1e-10
            print("First Frequency adjusted", self.freqBinSize * 1e-10, "MHz \
                  away from zero to avoid division errors.")
        elif self.first_freq < 0.0:
            raise ValueError("First Frequency Less Than Zero")
        self.last_freq = self.f0 + self.freqBinSize * self.Nf/2
        self.freq_Array = np.linspace(self.first_freq + self.freqBinSize/2,
                                      self.last_freq + self.freqBinSize/2,
                                      self.Nf, endpoint=False)

        if self.Nt*self.Nf > 500000:  # Limits the array size to 2.048 GB
            SignalPath = 'signal0.hdf5'
            if SignalType=='burst':  # Use a different file name for a burst
                SignalPath = "burst_signal.hdf5"

            if os.path.exists(SignalPath):
                if clean_mode:
                    os.remove(SignalPath)
                else:
                    ii = 1
                    while os.path.exists('signal{0}.hdf5'.format(ii)):
                        ii += 1
                    SignalPath = 'signal{0}.hdf5'.format(ii)

            SignalFile = h5py.File(SignalPath, 'a')
            self.signal = SignalFile.create_dataset(None, (rows, self.Nt),
                                                    dtype=self.data_type)
            #self.signal = np.memmap(SignalPath, dtype = self.data_type,
            #mode = 'w+', shape = (self.Nf, self.Nt))
        else:
            self.signal = np.zeros((rows, self.Nt), dtype=self.data_type)

        self.SignalDict['mode'] = mode
        self.MetaData.AddInfo(self.SignalDict)

    # Plotting Methods
    def pulse_plot(self, **kwargs):
        """Method to plot the pulse signal.

        Parameters
        ----------
        **kwargs
            For other keyword-only arguments, see pulse_plot() in PSS_plot.py

        Returns
        -------
        lines
            A pulse plot with specified axes. Y-axis can be intensity or voltage.
            X-axis can be time (ms) or phase.
        """
        return PSS_plot.pulse_plot(self, **kwargs)

    def filter_bank(self, **kwargs):
        """Method to produce filter bank plot

        Parameters
        ----------
        **kwargs
            For other keyword-only arguments, see filter_bank() in PSS_plot.py

        Returns
        -------
        lines
            A filter bank plot with specified X-axis, which can be a function of
            time (ms) or phase.
        """
        return PSS_plot.filter_bank(self, **kwargs)

    def profile_plot(self, **kwargs):
        """Method to plot the pulse profile template.

        Parameters
        ----------
        **kwargs
            For other keyword-only arguments, see profile_plot() in PSS_plot.py

        Returns
        -------
        lines
            A pulse profile template plot with specified axes. Y-axis can be intensity
            or voltage. X-axis can be time (ms) or phase.
        """
        return PSS_plot.profile_plot(self, **kwargs)

    #Set the signal parameters as properties and assign them to the MetaData

    @property
    def f0(self):
        """Property f0, central frequency (MHz)."""
        return self._f0

    @f0.setter
    def f0(self, value):
        self._f0 = value
        self.MetaData.f0 = self._f0

    @property
    def bw(self):
        """Property bw, bandwidth (MHz)."""
        return self._bw

    @bw.setter
    def bw(self, value):
        self._bw = value
        self.MetaData.bw = self._bw

    @property
    def Nf(self):
        """Property Nf, number of frequency bins."""
        return self._Nf

    @Nf.setter
    def Nf(self, value):
        self._Nf = value
        self.MetaData.Nf = self._Nf

    @property
    def Nt(self):
        """Property Nt, number of time/phase bins."""
        return self._Nt

    @Nt.setter
    def Nt(self, value):
        self._Nt = value
        self.MetaData.Nt = self._Nt

    @property
    def ObsTime(self):
        """Property ObsTime, total observation time in milliseconds."""
        return self._ObsTime

    @ObsTime.setter
    def ObsTime(self, value):
        self._ObsTime = value
        self.MetaData.ObsTime = self._ObsTime

    @property
    def data_type(self):
        """Property data_type, 'int8' or 'int16' supported.

        Automatically changed to 'uint8' or 'uint16' if intensity signal.
        """
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        self._data_type = value
        self.MetaData.data_type = self._data_type

    @property
    def SignalType(self):
        """Property SignalType, either 'intensity' or 'voltage'

        'intensity' carries a Nf x Nt filter bank of pulses or
        'voltage' which carries a 4 x Nt array of voltage vs. time pulses
        representing 4 stokes channels.
        """
        return self._SignalType

    @SignalType.setter
    def SignalType(self, value):
        self._SignalType = value
        self.MetaData.SignalType = self._SignalType
