
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy import stats

from ..utils.utils import make_quant

__all__ = ["Signal", "BaseSignal"]

class BaseSignal(object):
    """
    Base class for signals, subclass from this.

    Required Args:
        fcent [float]: central radio frequency

        bandwidth [float]: radio bandwidth of signal

    Optional Args:
        sample_rate [float]: sample rate of data, default: ``None``
            If no ``sample_rate`` is given the observation will default to
            the Nyquist frequency. Sub-Nyquist sampling is allowed, but a
            warning will be generated.

        dtype [type]: data type of array, default: ``np.float32``
            supported types are: ``np.float32`` and ``np.int8``

        Npols [int]: number of polarizations, 1-4. Currently only values of
                    1 for total intensity are supported.

    """

    _sigtype = "Signal"
    _Nchan = None
    _tobs = None  # set by Pulsar.make_pulses()

    _nsamp = None

    _draw_max = None
    _draw_norm = 1

    def __init__(self,
                 fcent, bandwidth,
                 sample_rate=None,
                 dtype=np.float32,
                 Npols=1):

        self._fcent = fcent
        # Check if bandwidth is negative; only an issue when making signal from fitsfile
        if bandwidth < 0:
            self._bw = np.abs(bandwidth)
        else:
            self._bw = bandwidth
        self._samprate = sample_rate
        if dtype is np.float32 or np.int8:
            self._dtype = dtype
        else:
            msg = "data type {} not supported".format(dtype)
            raise ValueError(msg)
        # TODO: This will be changed eventually to support multiple polarizations
        if Npols == 1:
            self._Npols = 1
        else:
            msg = "Only total intensity polarization is currently supported"
            raise ValueError(msg)

        # set total delay added to signal to be None
        self._delay = None

        self._dm = None

    def __repr__(self):
        return self.sigtype+"({0}, bw={1})".format(self.fcent, self.bw)

    def __add__(self, b):
        """overload ``+`` to concatinate signals"""
        raise NotImplementedError()

    def _set_draw_norm(self):
        """set distribution to correct dynamic range for int8

        This is only implemented for `numpy.int8` :class:`FilterBankSignals`.
        """
        raise NotImplementedError()

    def init_data(self, Nsamp):
        """
        Initialize a data array to store the signal.

        Required Args:
            Nsamp (int): number of data samples
        """
        self._data = np.empty((self.Nchan, Nsamp), dtype=self.dtype)

    def to_RF(self):
        """convert signal to RFSignal
        must be implemented in subclass!"""
        raise NotImplementedError()

    def to_Baseband(self):
        """convert signal to BasebandSignal
        must be implemented in subclass!"""
        raise NotImplementedError()

    def to_FilterBank(self, Nsubband=512):
        """convert signal to FilterBankSignal
        must be implemented in subclass!"""
        raise NotImplementedError()

    @property
    def data(self):
        return self._data

    @property
    def sigtype(self):
        return self._sigtype

    @property
    def Nchan(self):
        return self._Nchan

    @property
    def fcent(self):
        return self._fcent

    @property
    def bw(self):
        return self._bw

    @property
    def tobs(self):
            return self._tobs

    @property
    def samprate(self):
        return self._samprate
    
    @property
    def nsamp(self):
        return self._nsamp

    @property
    def dtype(self):
        return self._dtype

    @property
    def Npols(self):
        return self._Npols

    @property
    def dat_freq(self):
        return self._dat_freq

    @property
    def delay(self):
        return self._delay

    @property
    def dm(self):
        return self._dm

    @property
    def DM(self):
        return self._dm


def Signal():
    """helper function to instantiate signals
    """
    raise NotImplementedError()
