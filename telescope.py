"""telescope.py
module to add RFI and downsample data
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy as sp
from . import PSS_utils as utils

_kB = 1.38064852e+03  # Boltzmann const in radio units: Jy m^2 / K 

class Receiver(object):
    def __init__(self, centfreq, bandwidth, response=None, name=None):
        """Telescope Reciever"""
        self._name = name
        self._centfreq = centfreq
        self._bandwidth = bandwidth
        self._response = response

    def __repr__(self):
        return "Receiver({:s})".format(self._name)

    @property
    def name(self):
        return self._name
    @property
    def centfreq(self):
        return self._centfreq
    @property
    def bandwidth(self):
        return self._bandwidth
    @property
    def response(self):
        return self._response


class Backend(object):
    def __init__(self, samprate=None, name=None):
        self._name = name
        self._samprate = samprate

    def __repr__(self):
        return "Backend({:s})".format(self._name)

    @property
    def name(self):
        return self._name
    @property
    def samprate(self):
        return self._samprate

    def fold(self, signal, P_fold, N_fold=100):
        """fold(signal, P_fold, N_fold=100)
        fold a signal at period=P_fold, N_fold times
        """
        Nt, Nf = signal.shape
        Npbins = int(P_fold * 2*self.samprate)
        if Npbins*N_fold > Nt:
            print("UserWarning: not enough observed time for {:d} folds".format(N_fold))
            N_fold = Nt // Npbins
            w.warn("UserWarning: setting N_fold = {:d}".format(N_fold))
        fold_sig = signal[:,Npbins:Npbins*(N_fold+1)].reshape(Nf, N_fold, Npbins)
        return np.sum(fold_sig, axis=1)


class Telescope(object):
    """contains: observe(), noise(), rfi() methods"""
    def __init__(self, aperture, area=None, Tsys=None, name=None):
        """initalize telescope object
        aperture: aperture (m)
        area: collecting area (m^2) (if omitted, assume circular single dish)
        Tsys: system temp (K), total of receiver, sky, spillover, etc. (only needed for noise)
        name: string
        """
        #TODO: specify Trec in Receiver and compute others from pointing
        self._name = name
        self._area = area
        self._Tsys = Tsys
        self._aperture = aperture
        self._systems = {}
        if self._area is None:
            # assume circular single dish
            self._area = np.pi * (aperture/2)**2


    def __repr__(self):
        return "Telescope({:s}, {:f}m)".format(self._name, self._aperture)

    @property
    def name(self):
        return self._name
    @property
    def area(self):
        return self._area
    @property
    def Tsys(self):
        return self._Tsys
    @property
    def aperture(self):
        return self._aperture
    @property
    def systems(self):
        return self._systems

    def add_system(self, name=None, receiver=None, backend=None):
        """add_system(name=None, receiver=None, backend=None)
        append new system to dict systems"""
        self._systems[name] = (receiver, backend)

    def observe(self, signal, system=None, mode='search', noise=False):
        """observe(signal, system=None, mode='search', noise=False)
        signal -- Signal() instance
        system -- dict key for system to use
        """
        rec = self.systems[system][0]
        bak = self.systems[system][1]

        sig_in = signal.signal
        dt_tel = 1/(2*bak.samprate)
        dt_sig = signal.TotTime / signal.Nt

        if dt_sig == dt_tel:
            out = sig_in

        elif dt_tel % dt_sig == 0:
            SampFactor = int(dt_tel // dt_sig)
            new_Nt = int(signal.Nt//SampFactor)
            out = np.zeros((signal.Nf, new_Nt))
            for ii, row in enumerate(sig_in):
                out[ii,:] = utils.down_sample(row, SampFactor)
            print("Input signal sampling frequency= ", 1/dt_sig," kHz.\nTelescope sampling frequency = ",1/dt_tel," kHz")

        elif dt_tel > dt_sig:
            new_Nt = int(signal.TotTime // dt_tel)
            out = np.zeros((signal.Nf, new_Nt))
            for ii, row in enumerate(sig_in):
                out[ii,:] = utils.rebin(row, new_Nt)
            print("Input signal sampling frequency= ", dt_sig," ms. Telescope sampling frequency = ",dt_tel," ms")

        else:
            # Throw error if the input signal has a lower sampling frequency than the telescope sampling frequency.
            raise ValueError("Signal Sampling Frequency Lower than Telescope Sampling Frequency")

        if noise :
            out += self.radiometer_noise(signal, out.shape, dt_tel)
        
        if signal.SignalType == 'voltage':
            clip = signal.MetaData.gauss_draw_max

            out[out>clip] = clip
            out[out<-clip] = -clip
        else:
            clip = signal.MetaData.gamma_draw_max
            out[out>clip] = clip
        
        out = np.array(out, dtype=signal.MetaData.data_type)

        return out

    def radiometer_noise(self, signal, shape, dt):
        # flux density fluctuations: sigS from Lorimer & Kramer eq 7.12
        #TODO replace A with Aeff, depends on pointing for some telescopes
        #TODO Tsys -> Trec, compute Tsky, Tspill, Tatm from pointing
        dt *= 1.0e-3  # convert to sec
        BW = signal.bw  # MHz
        Np = signal.Npols
        G = self.area / (Np*_kB)  # K/Jy (gain)
        
        # noise variance
        sigS = self.Tsys / G / np.sqrt(Np * dt * BW)  # mJy
        
        if signal.SignalType == 'voltage':
            norm = np.sqrt(sigS) * signal.MetaData.gauss_draw_norm/signal.MetaData.Smax
            noise = norm * np.random.normal(0, 1, shape)
        else:
            norm = sigS * signal.MetaData.gamma_draw_norm/signal.MetaData.Smax
            noise = norm * np.random.chisquare(1, shape)

        return noise


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
    """
    g = Telescope(100.0, area=5500.0, Tsys=35.0, name="GBT")
    g.add_system(name="820_GUPPI",
                 receiver=Receiver(820, 180, name="820"),  # check me
                 backend=Backend(samprate=3.125, name="GUPPI"))
    g.add_system(name="Lband_GUPPI",
                 receiver=Receiver(1400, 400, name="Lband"),  # check me
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
                 receiver=Receiver(430, 100, name="430"),  # check me
                 backend=Backend(samprate=1.5625, name="PUPPI"))
    a.add_system(name="Lband_PUPPI",
                 receiver=Receiver(1410, 400, name="Lband"),  # check me
                 backend=Backend(samprate=12.5, name="PUPPI"))
    a.add_system(name="Sband_PUPPI",
                 receiver=Receiver(2030, 400, name="Sband"),  # check me
                 backend=Backend(samprate=12.5, name="PUPPI"))
    return a
