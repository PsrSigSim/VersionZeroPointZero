
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy import stats

from ..utils.utils import make_quant

__all__ = ['Receiver']


class Receiver(object):
    """
    Telescope receiver. A :class:`Receiver` must be instantiated with
    either a callable response function or ``fcent`` and ``bandwidth`` to use a
    flat response.

    Required Args:
        N/A

    Optional Args:
        response (callable): frequency response function ("bandpass") of
        receiver.

        fcent (float): center frequency of receiver with flat response (MHz)

        bandwidth (float): bandwidth of receiver with flat response (MHz)

        Trec (float): receiver temperature (K) for radiometer noise level,
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
                msg = "Non-flat response not yet implemented."
                raise NotImplementedError(msg)

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

    def radiometer_noise(self, signal, pulsar, gain=1, Tsys=None, Tenv=None):
        """
        Add radiometer noise to a signal.

        Tsys = Tenv + Trec, unless Tsys is given (just Trec if no Tenv)

        flux density fluctuations: sigS from Lorimer & Kramer eq 7.12
        """
        # Check if values have astropy units attached
        if hasattr(Tsys, 'value'):
            Tsys_check = Tsys.value
        else:
            Tsys_check = Tsys
        if hasattr(Tenv, 'value'):
            Tenv_check = Tenv.value
        else:
            Tenv_check = Tenv
            
        if Tsys_check is None and Tenv_check is None:
            Tsys = self.Trec
        elif Tenv_check is not None:
            if Tsys_check is not None:
                msg = "specify EITHER Tsys OR Tenv, not both"
                raise ValueError(msg)
            else:
                Tsys = Tenv + self.Trec
        # else: Tsys given as input!

        # gain by this equation should have units of K/Jy
        gain = make_quant(gain, "K/Jy")

        # select noise generation method
        if signal.sigtype in ["RFSignal", "BasebandSignal"]:
            noise = self._make_amp_noise(signal, Tsys, gain, pulsar)
        elif signal.sigtype == "FilterBankSignal":
            noise = self._make_pow_noise(signal, Tsys, gain, pulsar)
        else:
            msg = "no pulse method for signal: {}".format(signal.sigtype)
            raise NotImplementedError(msg)
        signal._data += noise

    def _make_amp_noise(self, signal, Tsys, gain, pulsar):
        """radiometer noise for amplitude signals
        """
        dt = 1 / signal.samprate

        # noise variance; hardcode npol = 2 for now (total intensity)
        sigS = Tsys / gain / np.sqrt(2 * dt * signal.bw)

        distr = stats.norm()

        U_scale = 1.0 / (np.sum(pulsar.Profiles._max_profile)/signal.samprate)

        norm = np.sqrt((sigS / signal._Smax).decompose())*U_scale
        noise = norm * distr.rvs(size=signal.data.shape)

        return noise.value  # drop units!

    def _make_pow_noise(self, signal, Tsys, gain, pulsar):
        """radiometer noise for power signals
        """
        # This definition is true regardless of mode for a filterbank signal
        nbins = signal.nsamp/signal.nsub # number of bins per subint
        dt = signal.sublen / nbins # bins per subint; s
        """
        if signal.fold:
            # number of bins per subint
            nbins = signal.nsamp/signal.nsub
            dt = signal.sublen / nbins # bins per subint; s
        else:
            nbins = signal.nsamp/signal.nsub
            dt = signal.sublen / nbins # bins per subint; s
            # Old definitions, depricated
            #nbins = signal.samprate
            #dt = 1 / nbins
        """
        bw_per_chan = signal.bw / signal.Nchan

        # noise variance; hardcode npol = 2 for now (total intensity)
        sigS = Tsys / gain / np.sqrt(2 * dt * bw_per_chan)

        df = signal.Nfold if signal.fold else 1
        distr = stats.chi2(df)

        # scaling factor due to profile normalization (see Lam et al. 2018a)
        U_scale = 1.0 / (np.sum(pulsar.Profiles._max_profile)/nbins)

        norm = (sigS * signal._draw_norm / signal._Smax).decompose() * U_scale
        noise = norm * distr.rvs(size=signal.data.shape)

        return noise.value  # drop units!



def response_from_data(fs, values):
    """generate a callable response function from discrete data
    Data are interpolated.
    """
    raise NotImplementedError()

def _flat_response(fcent, bandwidth):
    """
    Generate a callable, flat response function

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
