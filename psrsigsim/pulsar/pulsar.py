
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy import stats
from .profiles import GaussProfile
from .profiles import UserProfile
from .profiles import DataProfile
from ..utils.utils import make_quant, shift_t

class Pulsar(object):
    """class for pulsars

    The minimal data to instantiate a pulsar is the period, Smean, and
    pulse profile. The Profile is supplied via a :class:`PulseProfile`-like
    object.

    Parameters
    ----------

    period : float
        Pulse period (sec)

    Smean : float
        Mean pulse flux density (Jy)

    profile : :class:`PulseProfile`
        Pulse profile or 2-D pulse portrait

    name : str
        Name of pulsar
    """
    #TODO Other data could be supplied via a `.par` file.
    def __init__(self, period, Smean, profiles=None, name=None):
        self._period = make_quant(period, 's')
        self._Smean = make_quant(Smean, 'Jy')

        self._name = name

        # Assign profile class; default to GaussProfile if nothing is specified
        if profiles is None:
            self._Profiles = GaussProfile()
        else:
            self._Profiles = profiles

    def __repr__(self):
        namestr = "" if self.name is None else self.name+", "
        return "Pulsar("+namestr+"{})".format(self.period.to('ms'))

    @property
    def Profiles(self):
        return self._Profiles

    @property
    def name(self):
        return self._name

    @property
    def period(self):
        return self._period

    @property
    def Smean(self):
        return self._Smean

    def make_pulses(self, signal, tobs):
        """generate pulses from Profiles, :class:`PulsePortrait` object

        Required Args:
            signal (:class:`Signal`-like): signal object to store pulses
            tobs (float): observation time (sec)
        """
        signal._tobs = make_quant(tobs, 's')

        # init base profile at correct sample rate
        Nph = int((signal.samprate * self.period).decompose())

        # If there aren't enough frequencies in the profiles
        # And if it is a :class:`Profile` instance, reshape.
        self.Profiles.init_profiles(Nph, signal.Nchan)

        # if (signal.Nchan != self.Profiles.profiles.shape[0]
        #     and hasattr(self.Profiles, 'set_Nchan')):
        #     self.Profiles.set_Nchan(signal.Nchan)


        # select pulse generation method
        if signal.sigtype in ["RFSignal", "BasebandSignal"]:
            self._make_amp_pulses(signal)
        elif signal.sigtype == "FilterBankSignal":
            self._make_pow_pulses(signal)
        else:
            msg = "no pulse method for signal: {}".format(signal.sigtype)
            raise NotImplementedError(msg)

        # compute Smax (needed for radiometer noise level)
        #pr = self.Profiles()
        pr = self.Profiles._max_profile
        nbins = len(pr) # Think this assumes a single profile for now...
        signal._Smax = self.Smean * nbins / np.sum(pr)

    def _make_amp_pulses(self, signal):
        """generate amplitude pulses

        This method should be used for radio frequency and basebanded
        pulses.

        Parameters
        ----------

        signal : :class:`Signal`-like
            Signal object to store pulses.
        """
        # generate several pulses in time
        distr = stats.norm()

        signal._nsamp = int((signal.tobs * signal.samprate).decompose())
        signal.init_data(signal.nsamp)

        # TODO break into blocks
        # TODO phase from .par file
        # calc profile at phases
        phs = (np.arange(signal.nsamp) /
                (signal.samprate * self.period).decompose().value)
        phs %= 1  # clip integer part

        # convert intensity profile to amplitude!
        full_prof = np.sqrt(
            self.Profiles.calc_profiles(phs, Nchan=signal.Nchan)
        )

        signal._data = full_prof * distr.rvs(size=signal.data.shape)

    def _make_pow_pulses(self, signal):
        """generate a power pulse

        This method should be used for filter bank pulses

        Parameters
        ----------

        signal : :class:`Signal`-like
            Signal object to store pulses.
        """
        if signal.fold:
            # Determine how many subints to make
            if signal.sublen is None:
                signal._sublen = signal.tobs
                signal._nsub = 1
            else:
                # This should be an integer, if not, will round
                signal._nsub = int(np.round(signal.tobs / signal.sublen))

            # determine the number of data samples necessary
            signal._nsamp = int((signal.nsub*(self.period*signal.samprate)).decompose())
            # Need to make an initial empty data array
            signal.init_data(signal.nsamp)

            # Tile the profiles to number of desired subints
            sngl_prof = np.tile(self.Profiles(), signal.nsub)

            # changed to number of subints
            signal._Nfold = (signal.sublen / self.period).decompose()
            distr = stats.chi2(df=signal.Nfold)
            signal._set_draw_norm(df=signal.Nfold)

            #Why is there a second call to init_data?
            signal.init_data(sngl_prof.shape[1])
            signal._data = (sngl_prof * distr.rvs(size=signal.data.shape)
                            * signal._draw_norm)
        else:
            # fold is false and will make single pulses
            signal._sublen = self.period
            # This should be an integer, if not, will round; may not be exact
            signal._nsub = int(np.round((signal.tobs / signal.sublen).decompose()))

            # generate several pulses in time
            distr = stats.chi2(df=1)
            signal._set_draw_norm(df=1)

            signal._nsamp = int((signal.tobs * signal.samprate).decompose())
            signal.init_data(signal.nsamp)

            # TODO break into blocks
            # TODO phase from .par file
            # calc profiles at phases
            phs = (np.arange(signal.nsamp) /
                   (signal.samprate * self.period).decompose().value)
            phs %= 1  # clip integer part
            full_prof = self.Profiles.calc_profiles(phs,signal.Nchan)

            signal._data = (full_prof * distr.rvs(size=signal.data.shape)
                            * signal._draw_norm)

    def null(self, signal, null_frac, length=None, frequency=None):
        """
        Function to simulate pulsar pulse nulling. Given some nulling fraction,
        will replace simulated pulses with noise until nulling fraction is met.
        This function should only be run after running any ism or other delays
        have been added, e.g. disperison, FD, profile evolution, etc., but
        should be run before adding the radiometer noise ('telescope.observe()`),
        if nulling is desired.

        Parameters
        ----------

        signal [class] : signal class containing the simulated pulses
        null_frac [float] : desired pulsar nulling fraction, given as a
            decimal; range of 0.0 to 1.0.
        length [float] : desired length of each null in seconds. If not given,
                         will randomly null pulses. Default is None.
        frequency [float] : frequency of pulse nulling, e.g. how often the pulsar
                            nulls per hour. E.g. if frequency is 2, then the
                            pulsar will null twice per hour for some length
                            of time. If not given, will randomly null pulses.
                            Default is None.
        """
        # Determine how many pulses need to be nulled out
        null_pulses = int(np.round(signal.nsub * null_frac)) # needs to be int, may not be exact
        # Get the number of bins per pulse
        Nph = int((signal.samprate * self.period).decompose())
        # determine the off pulse window
        opw = self.Profiles._calcOffpulseWindow(Nphase = Nph)
        # define the random noise distribution
        if signal.fold:
            distr = stats.chi2(df=signal.Nfold)
        else:
            distr = stats.chi2(df=1)
        # have a test ditribution to determine null bins if Nfold is 1
        if not signal.fold or signal.Nfold < 100:
            check_distr = stats.chi2(df=100)
        else:
            check_distr = stats.chi2(df=signal.Nfold)
        # Figure out how off-center the pulses are
        shift_val = Nph//2 - np.where(signal.data[0,:Nph]==np.max(signal.data[0,:Nph]))[0]
        # Now null randomly if no length or frequency is given
        if length==None and frequency==None:
            # randomly draw pulse numbers to null
            rand_pulses = np.random.choice(signal.nsub, null_pulses, replace=False)
            # replace each pulse number with random noise
            if signal.delay == None:
                for p in rand_pulses:
                    # convert to bins
                    null_bins = np.arange(Nph*p, Nph*(p+1)) + shift_val
                    # make sure we don't go beyond the data array length
                    null_bins = null_bins[null_bins<np.shape(signal.data)[1]]
                    # replace pulses with noise
                    if len(null_bins) != Nph:
                        noise = (distr.rvs(size=len(null_bins)) * signal._draw_norm)
                    else:
                        noise = (distr.rvs(size=Nph) * signal._draw_norm)
                    # if no extra delays have been added to the signal
                    signal._data[:,null_bins] = noise * np.mean(self.Profiles._max_profile[opw.astype(int)])
            # if if signal has been delayed (e.g dispersed, etc.)
            else:
                # First initialize new array
                null_array = np.zeros(np.shape(signal.data))
                for p in rand_pulses:
                    #null_bins = np.arange(Nph*p, Nph*(p+1))
                    null_bins = np.arange(Nph*p, Nph*(p+1)) + shift_val
                    # make sure we don't go beyond the data array length
                    null_bins = null_bins[null_bins<np.shape(signal.data)[1]]
                    # replace the appropriate arrays
                    if len(null_bins) != Nph:
                        null_array[:,null_bins] = (check_distr.rvs(size=len(null_bins)) * signal._draw_norm)
                    else:
                        null_array[:,null_bins] = (check_distr.rvs(size=Nph) * signal._draw_norm)
                # now shift the data
                freq_array = signal._dat_freq
                shift_dt = (1/signal._samprate).to('ms')
                for ii, freq in enumerate(freq_array):
                    null_array[ii,:] = shift_t(null_array[ii,:],
                                                 signal.delay[ii].value,
                                                 dt=shift_dt.value)
                # Now replace pulses with noise
                off_pulse_mean = np.mean(self.Profiles._max_profile[opw.astype(int)])
                noise_shape = np.shape(np.where(null_array>1))[1]
                noise = (distr.rvs(size=noise_shape) * signal._draw_norm)
                signal._data[np.where(null_array>1)] = noise*off_pulse_mean

        else:
            raise NotImplementedError("Length and Frequency not been implimented yet")
