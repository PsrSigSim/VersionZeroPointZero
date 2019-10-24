
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
            self._area = np.pi * (aperture/2)**2
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

    def observe(self, signal, system=None, mode='search', noise=False):
        """observe(signal, system=None, mode='search', noise=False)
        signal -- Signal() instance
        system -- dict key for system to use
        """
        msg = "sig samp freq = {0:.3f} kHz\ntel samp freq = {1:.3f} kHz"
        rcvr = self.systems[system][0]
        bak = self.systems[system][1]

        sig_in = signal.data
        dt_tel = 1/(2*bak.samprate)
        # if we have subintegrations, need to get new dt_sig
        if signal.sublen:
            dt_sig = signal.sublen / (signal.nsamp/signal.nsub) # bins per subint; s
        else:
            dt_sig = signal.tobs / signal.nsamp # unit: s
        
        # QUESTION: out should be the resampled signal now right? So we 
        # should be editing the signal object directly here.
        
        if dt_sig == dt_tel:
            out = np.array(sig_in, dtype=float)

        elif dt_tel % dt_sig == 0:
            SampFactor = int(dt_tel // dt_sig)
            new_Nt = int(signal.nsamp//SampFactor)
            if signal.sigtype == 'voltage':
                out = np.zeros((signal.Npols, new_Nt))
            else:
                out = np.zeros((signal.Nchan, new_Nt))
            for ii, row in enumerate(sig_in):
                out[ii, :] = down_sample(row, SampFactor)
            print(msg.format((1/dt_sig).to("kHz").value, (1/dt_tel).to("kHz").value))

        elif dt_tel > dt_sig:
            new_Nt = int(signal.tobs // dt_tel)
            if signal.sigtype == 'voltage':
                out = np.zeros((signal.Npols, new_Nt))
            else:
                out = np.zeros((signal.Nchan, new_Nt))
            for ii, row in enumerate(sig_in):
                out[ii, :] = rebin(row, new_Nt)
            print(msg.format((1/dt_sig).to("kHz").value, (1/dt_tel).to("kHz").value))

        else:
            # input signal has lower samp freq than telescope samp freq
            #raise ValueError("Signal sampling freq < Telescope sampling freq")
            
            # circumvent this issue for subintegrated data for now...
            if signal.subint:
                out = np.array(sig_in, dtype=float)
            else:
                raise ValueError("Signal sampling freq < Telescope sampling freq")

        if noise:
            # The noise is getting added to the data in the radiometer noise function; this function as no output
            # Need to look into this resampling as well
            #out += rcvr.radiometer_noise(signal, gain=1) 
            rcvr.radiometer_noise(signal, gain=self.gain)
            

        if signal.sigtype == 'voltage':
            #clip = signal.MetaData.gauss_draw_max 
            # Difference between gauss and gamma draw here?
            clip = signal._draw_max

            out[out>clip] = clip
            out[out<-clip] = -clip
        else:
            #clip = signal.MetaData.gamma_draw_max
            # Difference between gauss and gamma draw here?
            clip = signal._draw_max
            out[out>clip] = clip

        out = np.array(out, dtype=signal.dtype)
        
        """
        What should out be? What is it taking the place of? How should we add noise to it?
        Currently appears to just be downsampled signal (which is fine but we need to know this)
        """
        return out

    def apply_response(self, signal):
        pass

    def rfi(self):
        pass

    def init_signal(self, system):
        """init_signal(system)
        instantiate a signal object with same Nt, Nf, bandwidth, etc
        as the system to be used for observation"""
        pass


# Convenience functions to construct GBT and AO telescopes
#TODO: should these be pre-instantiated?
#TODO: check Receivear centfreq & bandwidth
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
    return a
