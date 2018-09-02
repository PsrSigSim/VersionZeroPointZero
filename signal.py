"""pulsar.py
a starting point for the Pulsar class.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import h5py
from . import PSS_plot

class MetaData(object):
    """The MetaData class to contain information about the signal

    Attributes
    ----------
    f0 : int
        Central frequency of bandwidth (MHz)
    bw : int
        Bandwidth (MHz)
    Nf : int
        Number of frequency bins
    Nt : int
        Number of time/phase bins
    TotTime : int
        Total observation time in milliseconds
    """
    def __init__(self):
        self.f0 = None # central freq (MHz)
        self.bw = None # bandwidth (MHz)
        self.Nf = None # number of frequency bins
        self.Nt = None # number of time/phase bins
        self.TotTime = None #Total time in milliseconds

    def AddInfo(self,Info):
        """Function to add a dictionary of information into the metadata.

        Necessary since each new module will have a dictionary of terms to add in
        to the metadata

        Parameters
        ----------
        Info : dict
            A dictionary containing signal parameters to be added to metadata.
        """
        for ii,jj in Info.items():
            setattr(self, ii, jj)

        #TODO Add error message if someone trys to overlap parameters



class Signal(object):
    """The Signal class to c

    extended summary if necessary

    Attributes
    ----------
    f0 : int
        Central frequency (MHz).
    bw : int
        Bandwidth (MHz).
    Nf : int
        Number of frequency bins.
    Nt : int
        Number of time/phase bins.
    TotTime : int
        Total observation time in milliseconds.
    data_type : str
        Data type to store signal data with. 'int8' or 'int16' supported.
    SignalType : str
        Signal type (only either 'intensity' or 'voltage').
    mode : str
        Mode used to simulate effects. 'explore' to add effects individually.
        'simulation' to add selected effects at once.
    """
    def __init__(self, f0=1400, bw=400, Nf=20, Nt=1000, TotTime=200, data_type='int8', SignalType = "intensity", mode='explore'):
        """initialize Signal(), executed at assignment of new instance
        data_type = 'int8' or 'int16' supported.
                    Automatically changed to 'uint8' or 'uint16' if intensity signal.
        @param f0 -- central frequency (MHz)
        @param bw -- bandwidth (MHz)
        @param Nf -- number of freq. bins
        @param Nt -- number of phase bins
        Totime = total time of the observation in milliseconds
        SignalType = 'intensity' which carries a Nf x Nt filterbank of pulses or 'voltage'
                    which carries a 4 x Nt array of voltage vs. time pulses representing 4 stokes channels
        """
        self.MetaData = MetaData()
        self.f0 = f0 # (MHz)
        self.bw = bw # (MHz)
         # freq bins
        self.data_type = data_type
        self.SignalType = SignalType
        self.SignalDict = {}
        if Nt%2 == 0: # Make signal even in length (for FFTs)
            self.Nt = Nt # phase bins
        else: self.Nt = Nt + 1

        if SignalType == 'voltage' and data_type == 'int8':
            self.data_type = 'int8'
            rows = 4 #Easy way to make 2 complex channels of voltage
            self.Npols = int(4)
            self.Nf = int(Nf)
            self.SignalDict['gauss_draw_max'] = np.iinfo(np.int8).max
            self.SignalDict['data_type'] = np.int8
            # Set the correct standard deviation for the
            # pulse draws depending on data type.

        elif SignalType == 'voltage' and data_type == 'int16':
            self.data_type = 'int16'
            rows = int(4) #Easy way to make 2 complex channels of voltage
            self.Npols = int(4)
            self.Nf = int(Nf)
            self.SignalDict['gauss_draw_max'] = np.iinfo(np.int16).max
            self.SignalDict['data_type'] = np.int16

        elif SignalType == 'intensity' and data_type == 'int8':
            self.data_type = 'uint8'
            self.Nf = int(Nf)
            self.Npols = int(1)
            rows = self.Nf
            self.SignalDict['gamma_draw_max'] = np.iinfo(np.uint8).max
            self.SignalDict['data_type'] = np.uint8

        elif SignalType == 'intensity' and data_type == 'int16':
            self.data_type = 'uint16'
            self.Nf = int(Nf)
            self.Npols = int(1)
            rows = self.Nf
            self.SignalDict['gamma_draw_max'] = np.iinfo(np.uint16).max
            self.SignalDict['data_type'] = np.uint16

        else:
            raise ValueError('Signal Type: '+SignalType+' and data type: '+data_type+' not supported together.')

        self.TotTime = TotTime #Total time in milliseconds
        self.TimeBinSize = self.TotTime/self.Nt
        self.freqBinSize = self.bw/self.Nf
        self.first_freq = self.f0 - self.freqBinSize * self.Nf/2
        if self.first_freq == 0.0 :
            self.first_freq = self.first_freq + self.freqBinSize * 1e-10
            print("First Frequency adjusted",self.freqBinSize * 1e-10,"MHz away from zero to avoid division errors.")
        elif self.first_freq < 0.0 :
            raise ValueError("First Frequency Less Than Zero")
        self.last_freq = self.f0 + self.freqBinSize * self.Nf/2
        self.freq_Array = np.linspace(self.first_freq + self.freqBinSize/2, self.last_freq + self.freqBinSize/2, self.Nf, endpoint=False)

        if self.Nt*self.Nf > 500000: #Limits the array size to 2.048 GB
            SignalPath = "signal.hdf5"
            if SignalType=='burst': #Use a different file name for a burst
                SignalPath = "burst_signal.hdf5"
            SignalFile = h5py.File(SignalPath,'a')
            self.signal = SignalFile.create_dataset(None, (rows, self.Nt), dtype=self.data_type)
            #self.signal = np.memmap(SignalPath, dtype = self.data_type, mode = 'w+', shape = (self.Nf, self.Nt))
        else:
            self.signal = np.zeros((rows, self.Nt),dtype=self.data_type)

        self.SignalDict['mode'] = mode
        self.MetaData.AddInfo(self.SignalDict)

    ### Plotting Methods
    def pulse_plot(self, **kwargs):
        return PSS_plot.pulse_plot(self, **kwargs)

    def filter_bank(self, **kwargs):
        return PSS_plot.filter_bank(self, **kwargs)

    def profile_plot(self, **kwargs):
        return PSS_plot.profile_plot(self, **kwargs)

    #Set the signal parameters as properties and assign them to the MetaData

    @property

    def f0(self):
        return self._f0

    @f0.setter
    def f0(self, value):
        self._f0 = value
        self.MetaData.f0 = self._f0

    @property

    def bw(self):
        return self._bw

    @bw.setter
    def bw(self, value):
        self._bw = value
        self.MetaData.bw = self._bw

    @property

    def Nf(self):
        return self._Nf

    @Nf.setter
    def Nf(self, value):
        self._Nf = value
        self.MetaData.Nf = self._Nf

    @property

    def Nt(self):
        return self._Nt

    @Nt.setter
    def Nt(self, value):
        self._Nt = value
        self.MetaData.Nt = self._Nt

    @property

    def TotTime(self):
        return self._TotTime

    @TotTime.setter
    def TotTime(self, value):
        self._TotTime = value
        self.MetaData.TotTime = self._TotTime

    @property

    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        self._data_type = value
        self.MetaData.data_type = self._data_type

    @property

    def SignalType(self):
        return self._SignalType

    @SignalType.setter
    def SignalType(self, value):
        self._SignalType = value
        self.MetaData.SignalType = self._SignalType
