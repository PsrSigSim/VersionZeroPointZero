
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy import stats

from ..utils.utils import make_quant

__all__ = ['Receiver']


class Receiver(object):
    """telescope reciever

    A :class:`Receiver` must be instatiated with either a callable response
    function or ``fcent`` and ``bandwidth`` to use a flat response.

    Optional Args:
        response (callable): frequency response function ("bandpass") of
            receiver.
        fcent (float): center frequency of reciever with flat response (MHz)
        bandwidth (float): bandwidth of reciever with flat response (MHz)
        Trec (float): reciever temperature (K) for radiometer noise level,
            default: ``35``
    """
    def __init__(self,
                 response=None, 
                 fcent=None, bandwidth=None,
                 Trec=35,
                 name=None):
        if response is None: 
            if fcent is None or bandwidth is None:
                msg = "specify EITHER response OR fcent and bandwidth"
                raise ValueError(msg)
            else:
                self._response = _flat_response(fcent, bandwidth)
        else:
            if fcent is not None or bandwidth is not None:
                msg = "specify EITHER response OR fcent and bandwidth"
                raise ValueError(msg)
            else:
                self._response = response

        self._Trec = make_quant(Trec, "K")
        self._name = name
        self._fcent = make_quant(fcent, "MHz")
        self._bandwidth = make_quant(bandwidth, "MHz")

    def __repr__(self):
        return "Receiver({:s})".format(self._name)

    @property
    def name(self):
        return self._name

    @property
    def Trec(self):
        return self._Trec

    @property
    def response(self):
        return self._response
    
    @property
    def fcent(self):
        return self._fcent
    
    @property
    def bandwidth(self):
        return self._bandwidth

    def radiometer_noise(self, signal, gain=1, Tsys=None, Tenv=None):
        """add radiometer noise to a signal

        Tsys = Tenv + Trec, unless Tsys is given (just Trec if no Tenv)
        
        flux density fluctuations: sigS from Lorimer & Kramer eq 7.12
        """
        if Tsys is None and Tenv is None:
            Tsys = self.Trec
        elif Tenv is not None:
            if Tsys is not None:
                msg = "specify EITHER Tsys OR Tenv, not both"
                raise ValueError(msg)
            else:
                Tsys = Tenv + self.Trec
        # else: Tsys given as input!
        
        # gain by this equation should have units of K/Jy
        gain = make_quant(gain, "K/Jy")

        # select noise generation method
        if signal.sigtype in ["RFSignal", "BasebandSignal"]:
            noise = self._make_amp_noise(signal, Tsys, gain)
        elif signal.sigtype is "FilterBankSignal":
            noise = self._make_pow_noise(signal, Tsys, gain)
        else:
            msg = "no pulse method for signal: {}".format(signal.sigtype)
            raise NotImplementedError(msg)
        signal._data += noise

    def _make_amp_noise(self, signal, Tsys, gain):
        """radiometer noise for amplitude signals
        """
        dt = 1 / signal.samprate

        # noise variance
        sigS = Tsys / gain / np.sqrt(dt * signal.bw)

        distr = stats.norm()

        norm = np.sqrt((sigS / signal._Smax).decompose())
        noise = norm * distr.rvs(size=signal.data.shape)
        
        return noise.value  # drop units!

    def _make_pow_noise(self, signal, Tsys, gain):
        """radiometer noise for power signals
        """
        if signal.subint:
            dt = signal.sublen / (signal.nsamp/signal.nsub) # bins per subint; s
        else:
            dt = 1 / signal.samprate
        bw_per_chan = signal.bw / signal.Nchan

        # noise variance
        sigS = Tsys / gain / np.sqrt(dt * bw_per_chan)

        df = signal.Nfold if signal.subint else 1
        distr = stats.chi2(df)

        norm = (sigS * signal._draw_norm / signal._Smax).decompose()
        noise = norm * distr.rvs(size=signal.data.shape)

        return noise.value  # drop units!



def response_from_data(fs, values):
    """generate a callable response function from discrete data
    Data are interpolated.
    """
    raise NotImplementedError()

def _flat_response(fcent, bandwidth):
    """generate a callable, flat response function
    Required Args:
        fcent (float): central frequency (MHz)
        bandwidth (float): bandwidth (MHz)
    Returns:
        callable bandpass(f), where f is an array or scalar frequency (MHz)
    """
    fc = make_quant(fcent, 'MHz')
    bw = make_quant(bandwidth, 'MHz')
    fmin = fc - bw/2
    fmax = fc + bw/2
    return lambda f: np.heaviside(f-fmin, 0) * np.heaviside(fmax-f, 0)

