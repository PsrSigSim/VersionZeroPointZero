
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np

from .receiver import Receiver, _flat_response, response_from_data
from .backend import Backend
from ..utils.utils import make_quant, down_sample, rebin

__all__ = ['Telescope', 'GBT', 'Arecibo']

_kB = make_quant(1.38064852e+03, "Jy*m^2/K")  # Boltzmann const in radio units

class Telescope(object):
    """contains: observe(), noise(), rfi() methods"""
    def __init__(self, aperture, area=None, Tsys=None, name=None):
        """initalize telescope object
        aperture: aperture (m)
        area: collecting area (m^2) (if omitted, assume circular single dish)
        Tsys: System temperature (K) of the telescope (if omitted use Trec)
        name: string
        """ # noqa E501
        #TODO: specify Trec in Receiver and compute others from pointing
        self._name = name
        self._aperture = make_quant(aperture, "m")
        self._systems = {}

        if area is None:
            # assume circular single dish
            self._area = np.pi * (self.aperture/2)**2
        else:
            self._area = make_quant(area, "m^2")
        self._gain = self.area / (2*_kB)  # 2 polarizations

        if Tsys == None:
            self._Tsys = Tsys
        else:
            self._Tsys = make_quant(Tsys, "K")

    def __repr__(self):
        return "Telescope({:s}, {:f}m)".format(self._name, self._aperture)

    @property
    def name(self):
        return self._name

    @property
    def area(self):
        return self._area

    @property
    def gain(self):
        return self._gain

    @property
    def aperture(self):
        return self._aperture

    @property
    def systems(self):
        return self._systems

    @property
    def Tsys(self):
        return self._Tsys

    def add_system(self, name=None, receiver=None, backend=None):
        """add_system(name=None, receiver=None, backend=None)
        append new system to dict systems"""
        self._systems[name] = (receiver, backend)

    def observe(self, signal, pulsar, system=None, noise=False, ret_resampsig = False):
        """
        Parameters
        ----------

        signal -- Signal() instance
        pulsar -- Pulsar() object, necessary for radiometer noise scaling
        system -- dict key for system to use
        noise : bool
            If True will add radiometer noise to the signal data.
        ret_resampsig : bool
            If True will return the resampled signal as a numpy array. Otherwise
            will not return anything.
        """
        if signal.sigtype in ["RFSignal", "BasebandSignal"]:
            raise NotImplementedError

        msg = "sig samp freq = {0:.3f} kHz\ntel samp freq = {1:.3f} kHz"
        rcvr = self.systems[system][0]
        bak = self.systems[system][1]

        sig_in = signal.data
        dt_tel = 1/(2*bak.samprate)
        # if we have subintegrations, need to get new dt_sig
        if (signal.sigtype == "FilterBankSignal"
            and signal.sublen is not None):
            dt_sig = signal.sublen / (signal.nsamp/signal.nsub)
        else:
            dt_sig = signal.tobs / signal.nsamp

        if dt_sig == dt_tel:
            out = np.array(sig_in, dtype=float)

        elif dt_tel % dt_sig == 0:
            SampFactor = int(dt_tel // dt_sig)
            new_Nt = int(signal.nsamp//SampFactor)
            out = np.zeros((signal.Nchan, new_Nt))
            for ii, row in enumerate(sig_in):
                out[ii, :] = down_sample(row, SampFactor)
            print(msg.format((1/dt_sig).to("kHz").value, (1/dt_tel).to("kHz").value))

        elif dt_tel > dt_sig:
            new_Nt = int(signal.tobs // dt_tel)
            if signal.sigtype in ["RFSignal", "BasebandSignal"]:
                out = np.zeros((signal.Nchan, new_Nt))
            else:
                out = np.zeros((signal.Nchan, new_Nt))
            for ii, row in enumerate(sig_in):
                out[ii, :] = rebin(row, new_Nt)
            print(msg.format((1/dt_sig).to("kHz").value, (1/dt_tel).to("kHz").value))

        else:
            # input signal has lower samp freq than telescope samp freq
            # We will need to fix this eventually but for now we will circumvent
            out = np.array(sig_in, dtype=float)

        if noise:
            # The noise is getting added to the data in the radiometer noise function; this function as no output
            # Need to look into this resampling as well
            rcvr.radiometer_noise(signal, pulsar,
                                  gain=self.gain, Tsys=self.Tsys)

        if signal.sigtype in ["RFSignal", "BasebandSignal"]:
            # Difference between gauss and gamma draw here?
            clip = signal._draw_max

            out[out>clip] = clip
            out[out<-clip] = -clip
        else:
            # Difference between gauss and gamma draw here?
            clip = signal._draw_max
            out[out>clip] = clip

        out = np.array(out, dtype=signal.dtype)

        if ret_resampsig:
            # Return the downsampled signal
            return out

    def apply_response(self, signal):
        raise NotImplementedError()

    def rfi(self):
        raise NotImplementedError()

    def init_signal(self, system):
        """init_signal(system)
        instantiate a signal object with same Nt, Nf, bandwidth, etc
        as the system to be used for observation"""
        raise NotImplementedError()


# Convenience functions to construct GBT and AO telescopes
#TODO: should these be pre-instantiated?
#TODO: check Receivear centfreq & bandwidth

"""
About G/ASP and XUPPI backend sampling rates and bandwidths. May need to change
some of these evetually...
From Paul Demorest:

so both instruments work by sampling a "wide" band then digitally filtering into a number of smaller channels,
with reduced rate per-channel.  do you need the original rate, or the per-channel rate?

anyways, for GUPPI/PUPPI the original sampling rate is either 1.6 GHz, 400 MHz,
or 200 MHz depending on which total BW mode was in use (800, 200, or 100 MHz), we
use different modes for different receivers.

for ASP/GASP original sample rate was fixed at 128 MHz.

The per-channel rate in all cases is equal to the channel BW (because a complex
data representation was used).  4 MHz for ASP/GASP and 1.5625 for GUPPI/PUPPI for nanograv obs.
"""

def GBT():
    """The 100m Green Bank Telescope
    at ~1 GHz: effective area ~ 5500 m^2
               Tsys ~ 35 K
    see: http://www.gb.nrao.edu/~rmaddale/GBT/ReceiverPerformance/PlaningObservations.htm
    """ # noqa E501
    g = Telescope(100.0, area=5500.0, Tsys=35.0, name="GBT")
    g.add_system(name="820_GUPPI",
                 receiver=Receiver(fcent=820, bandwidth=180, name="820"),  # check me
                 backend=Backend(samprate=3.125, name="GUPPI"))
    g.add_system(name="Lband_GUPPI",
                 receiver=Receiver(fcent=1400, bandwidth=800, name="Lband"),  # check me
                 backend=Backend(samprate=12.5, name="GUPPI"))
    # Parameters from NANOGrav 9 year paper
    g.add_system(name="800_GASP",
                 receiver=Receiver(fcent=844, bandwidth=64, name="800"),  # check me
                 backend=Backend(samprate=0.25, name="GASP"))
    g.add_system(name="Lband_GASP",
                 receiver=Receiver(fcent=1410, bandwidth=64, name="Lband"),  # check me
                 backend=Backend(samprate=0.25, name="GASP"))
    return g


def Arecibo():
    """The Arecibo 300m Telescope
    with Lwide: effective area ~ 22000 m^2 (G~10)
                Tsys ~ 35 K
    see: http://www.naic.edu/~astro/RXstatus/rcvrtabz.shtml
    """
    a = Telescope(300.0, area=22000.0, Tsys=35.0, name="Arecibo")
    a.add_system(name="430_PUPPI",
                 receiver=Receiver(fcent=430, bandwidth=100, name="430"),  # check me
                 backend=Backend(samprate=1.5625, name="PUPPI"))
    a.add_system(name="Lband_PUPPI",
                 receiver=Receiver(fcent=1410, bandwidth=800, name="Lband"),  # check me
                 backend=Backend(samprate=12.5, name="PUPPI"))
    a.add_system(name="Sband_PUPPI",
                 receiver=Receiver(fcent=2030, bandwidth=400, name="Sband"),  # check me
                 backend=Backend(samprate=12.5, name="PUPPI"))
    # Parameters from NANOGrav 9 year paper
    a.add_system(name="327_ASP",
                 receiver=Receiver(fcent=327, bandwidth=64, name="327"),  # check me
                 backend=Backend(samprate=0.25, name="ASP"))
    a.add_system(name="430_ASP",
                 receiver=Receiver(fcent=432, bandwidth=64, name="430"),  # check me
                 backend=Backend(samprate=0.25, name="ASP"))
    a.add_system(name="Lband_ASP",
                 receiver=Receiver(fcent=1412, bandwidth=64, name="Lband"),  # check me
                 backend=Backend(samprate=0.25, name="ASP"))
    a.add_system(name="Sband_ASP",
                 receiver=Receiver(fcent=2348, bandwidth=64, name="Sband"),  # check me
                 backend=Backend(samprate=0.25, name="ASP"))

    return a
