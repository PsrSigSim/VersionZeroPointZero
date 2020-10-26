
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np

from .signal import BaseSignal
from ..utils.utils import make_quant

class BasebandSignal(BaseSignal):
    """
    A time domain base-banded signal.

    A :class:`BasebandSignal` covers frequencies from 0 Hz to its bandwidth,
    e.g. ~1 GHz for L-band GUPPI.  In reality the telescope backend basebands
    the RF signal, but we allow pre-basebanded signals to save memory.
    
    Required Args:
        fcent [float]: central radio frequency (MHz)

        bandwidth [float]: radio bandwidth of signal (MHz)

    Optional Args:
        sample_rate [float]: sample rate of data (MHz), default: ``None``
            If no ``sample_rate`` is given the observation will default to
            the Nyquist frequency. Sub-Nyquist sampling is allowed, but a
            warning will be generated.

        dtype [type]: data type of array, default: ``np.float32``
            any numpy compatible floating point type will work
            
        Nchan [int]: number of polarization channels to simulate, default is 2.
    """

    _sigtype = "BasebandSignal"

    def __init__(self,
                 fcent, bandwidth,
                 sample_rate=None,
                 dtype=np.float32,
                 Nchan = 2):

        self._fcent = make_quant(fcent, 'MHz')
        self._bw = make_quant(bandwidth, 'MHz')
        self._Nchan = Nchan

        f_Nyquist = 2 * self._bw
        if sample_rate is None:
            self._samprate = f_Nyquist
        else:
            self._samprate = make_quant(sample_rate, 'MHz')
            if self._samprate < f_Nyquist:
                msg = ("specified sample rate {} < Nyquist frequency {}"
                       .format(self._samprate, f_Nyquist))
                print("Warning: "+msg)

        self._dtype = dtype

    def to_RF(self):
        """convert signal to RFSignal
        """
        # rfft this signal
        # generate zero freq domain array for RFSignal
        # put this signal into correct span of RFSignal
        # irfft RFSignal
        raise NotImplementedError()

    def to_Baseband(self):
        """convert signal to BasebandSignal
        """
        return self

    def to_FilterBank(self, Nsubband=512):
        """convert signal to FilterBankSignal
        """
        # TODO
        raise NotImplementedError()
