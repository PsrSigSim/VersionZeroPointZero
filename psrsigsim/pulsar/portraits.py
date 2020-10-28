
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from astropy import log
from scipy.interpolate import CubicSpline as _cubeSpline


class PulsePortrait(object):
    """
    Base class for pulse profiles.

    A pulse portrait is a set of profiles across the frequency range. They are
    INTENSITY series, even when using amplitude style signals
    (like :class:`BasebandSignal` ).
    """
    _profiles = None

    def __call__(self, phases=None):
        """
        A :class:`PulsePortrait` returns the actual profiles calculated
        at the specified phases when called
        """
        if phases is None:
            if self._profiles is None:
                msg = "base profiles not generated, returning `None`"
                print("Warning: "+msg)
            return self._profiles
        else:
            return self.calc_profiles(phases)

    def init_profiles(self, Nphase, Nchan=None):
        """
        Generate the profile, evenly sampled.

        Parameters
        ----------

            Nphase (int): number of phase bins
        """
        ph = np.arange(Nphase)/Nphase
        self._profiles = self.calc_profiles(ph, Nchan=Nchan)
        self._Amax = self._profiles.max()
        self._profiles /= self.Amax
        self._max_profile = [pr for pr in self._profiles if pr.max()==1.0][0]

    def calc_profiles(self, phases, Nchan=None):
        """
        Calculate the profiles at specified phase(s)

        Args:
            phases (array-like): phases to calc profile
        Note:
            The normalization can be wrong, if you have not run
            ``init_profiles`` AND you are generating less than one
            rotation.

        This is implemented by the subclasses!
        """
        raise NotImplementedError()

    def _calcOffpulseWindow(self, Nphase = None):
        """
        Function adapted from Pypulse (https://github.com/mtlam/PyPulse)
        to determine the offpulse window of the input profile
        """
        # Find minimum in the area
        if Nphase == None:
            windowsize = 2048/8
        else:
            windowsize = Nphase/8

        bins = np.arange(0, Nphase)

        integral = np.zeros_like(self._max_profile)
        for i in bins:
            win = np.arange(i-windowsize//2, i+windowsize//2) % Nphase
            integral[i] = np.trapz(self._max_profile[win.astype(int)])
        minind = np.argmin(integral)
        opw = np.arange(minind-windowsize//2, minind+windowsize//2+1)
        opw = opw % Nphase
        return opw

    @property
    def profiles(self):
        return self._profiles

    @property
    def Amax(self):
        return self._Amax



class GaussPortrait(PulsePortrait):
    """
    Sum of gaussian components.

    The shape of the inputs determine the number of gaussian components
    in the pulse.
    single float  : Single pulse profile made of a single gaussian

    1-d array     : Single pulse profile made up of multiple gaussians
    where `n` is the number of Gaussian components in the profile.

    Parameters
    ----------

    peak : float)
        Center of gaussian in pulse phase.

    width : float
        Stdev of pulse in pulse phase, default: ``0.1``

    amp : float
        Amplitude of pulse relative to other pulses, `default: ``1``

    Profile is renormalized so that maximum is 1.
    See the `Pulsar._make_amp_pulses()`, `Pulsar._make_pow_pulses()`, and
    `pPlsar.make_pulses()` methods for more details.
    """

    def __init__(self, peak=0.5, width=0.05, amp=1):
        #TODO: error checking for array length consistency?
        #TODO: if any param is a not array, then broadcast to all entries of other arrays?

        self._peak = peak
        self._width = width
        self._amp = amp
        self._profiles = None

    def init_profiles(self, Nphase, Nchan=None):
        """
        Generate the profile.

        Args:
            Nphase (int): number of phase bins
        """
        ph = np.arange(Nphase)/Nphase
        self._profiles = self.calc_profiles(ph, Nchan=Nchan)
        self._max_profile = [pr for pr in self._profiles if pr.max()==1.0][0]


    def calc_profiles(self, phases, Nchan=None):
        """
        Calculate the profiles at specified phase(s).

        Args:
            phases (array-like): phases to calc profile
        Note:
            The normalization can be wrong, if you have not run
            ``init_profile`` AND you are generating less than one
            rotation.
        """
        ph = np.array(phases)
        # profile = np.zeros_like(ph)

        if hasattr(self.peak,'ndim'):
            if self.peak.ndim == 1:
                if Nchan is None:
                    err_msg = 'Nchan must be provided if only 1-dim profile '
                    err_msg += 'information provided.'
                    raise ValueError(err_msg)
                profile = _gaussian_mult_1d(ph, self.peak, self.width, self.amp)
                profiles = np.tile(profile,(Nchan,1))
            elif self.peak.ndim == 2:
                Nchan = self.peak.shape[0]
                profiles = _gaussian_mult_2d(ph, self.peak, self.width,
                                                   self.amp, Nchan)
        else:
            if Nchan is None:
                err_msg = 'Nchan must be provided if only 1-dim profile '
                err_msg += 'information provided.'
                raise ValueError(err_msg)
            profile = _gaussian_sing_1d(ph, self.peak, self.width, self.amp)
            profiles = np.tile(profile,(Nchan,1))

        self._Amax = self.Amax if hasattr(self, '_Amax') else np.amax(profiles)
        return profiles / self._Amax

    @property
    def profiles(self):
        return self._profiles

    @property
    def peak(self):
        return self._peak

    @property
    def width(self):
        return self._width

    @property
    def amp(self):
        return self._amp

    @property
    def Amax(self):
        return self._Amax

class DataPortrait(PulsePortrait):
    """
    A pulse portrait generated from data.

    The data are samples of the profiles at specified phases. If you have a
    functional form for the _profiles use :class:`UserProfile` instead.

    Parameters
    ----------

    profiles : array, list of lists
        Profile data in 2-d array.


    phases : array, list of lists (optional)
        List of sampled phases. If phases are omitted profile is assumed to be
        evenly sampled and cover one whole rotation period.

    Profile is renormalized so that maximum is 1.
    See the `Pulsar._make_amp_pulses()`, `Pulsar._make_pow_pulses()`, and
    `Pulsar.make_pulses()` methods for more details.
    """
    def __init__(self, profiles, phases=None):
        # Check that no profile bins are below zero intensity
        if np.any(profiles < 0.0):
            log.warning("Some phase bins of input profile are negative, replacing them with zeros...")
            for prof in profiles:
                if np.any(prof < 0.0):
                    neg_idxs = np.where(prof < 0.0)[0]
                    prof[neg_idxs] = 0.0
        
        if phases is None:
            # infer phases
            N = profiles.shape[1]
            if any([ii != jj for ii,jj in zip(profiles[:,0], profiles[:,-1])]):
                # enforce periodicity!
                profiles = np.append(profiles, profiles[:,0][:,np.newaxis],
                                     axis=1)
                phases = np.arange(N+1)/N
            else:
                phases = np.arange(N)/N
        else:
            if phases[-1] != 1:
                # enforce periodicity!
                phases = np.append(phases, 1)
                profiles = np.append(profiles, profiles[:,0][:,np.newaxis],
                                     axis=1)
            elif any([ii != jj for ii,jj in zip(profiles[:,0],
                                                profiles[:,-1])]):
                # enforce periodicity!
                profiles[:,-1] = profiles[:,0]

        self._generator = _cubeSpline(phases, profiles, axis=1,
                                      bc_type='periodic')

    def calc_profiles(self, phases, Nchan=None):
        """
        Calculate the profile at specified phase(s).

        Args:
            phases (array-like): phases to calc profile
        Note:
            The normalization can be wrong, if you have not run
            ``init_profile`` AND you are generating less than one
            rotation.
        """
        profiles = self._generator(phases)
        Amax = self.Amax if hasattr(self, '_Amax') else np.max(profiles)
        return profiles / Amax


class UserPortrait(PulsePortrait):
    """
    User specified 2-D pulse portrait.
    """
    def __init__(self):
        raise NotImplementedError()

def _gaussian_sing_1d(phases, peak, width, amp):
    """Plot a single 1-dim gaussian."""
    if any(phases>1) or any(phases<0):
        raise ValueError('Phase values must all lie within [0,1].')
    return (amp * np.exp(-0.5 * ((phases-peak)/width)**2))

def _gaussian_mult_1d(phases, peaks, widths, amps):
    """Plot a 1-dim sum of multiple gaussians."""
    if any(phases>1) or any(phases<0):
        raise ValueError('Phase values must all lie within [0,1].')

    prof = (amps[:, np.newaxis] * np.exp(-0.5 * ((phases[np.newaxis,:]
                                         -peaks[:,np.newaxis])
                                         /widths[:,np.newaxis])**2))
    return np.sum(prof, axis=0)

def _gaussian_mult_2d(phases, peaks, widths, amps, nchan):
    return np.array([_gaussian_mult_1d(phases, peaks[:],
                                       widths[:], amps[:])
                                       for ii in range(nchan)])
