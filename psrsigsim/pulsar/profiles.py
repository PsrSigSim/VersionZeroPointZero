
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from ..utils.utils import make_quant
from .portraits import PulsePortrait, GaussPortrait, DataPortrait, UserPortrait
from astropy import log
from scipy.interpolate import CubicSpline as _cubeSpline

class PulseProfile(PulsePortrait):
    """
    Base class for pulse profiles

    Pulse profiles are INTENSITY series, even when using amplitude style
    signals (like :class:`BasebandSignal` ).
    """
    _profile = None

    def __call__(self, phases=None):
        """
        A :class:`PulseProfile` returns the actual profile calculated
        at the specified phases when called.
        """
        if phases is None:
            if self._profile is None:
                msg = "base profile not generated, returning `None`"
                print("Warning: "+msg)
            return self._profile
        else:
            return self.calc_profile(phases)

    def init_profile(self, Nphase):
        """
        Generate the profile, evenly sampled.

        Args:
            Nphase (int): number of phase bins
        """
        ph = np.arange(Nphase)/Nphase
        self._profile = self.calc_profile(ph)
        self._Amax = self._profile.max()
        self._profile /= self.Amax

    def calc_profile(self, phases):
        """
        Calculate the profile at specified phase(s).
        This is implemented by the subclasses!

        Args:
            phases (array-like): phases to calc profile

        Note:
            The normalization can be wrong, if you have not run
            ``init_profile`` AND you are generating less than one
            rotation.
        """
        raise NotImplementedError()

    @property
    def profile(self):
        return self._profile

    @property
    def Amax(self):
        return self._Amax


class GaussProfile(GaussPortrait):
    """
    Sum of guassian components.

    The shape of the inputs determine the number of gaussian components
    in the pulse.

    single float  : Single pulse profile made of a single gaussian

    1-d array     : Single pulse profile made up of multiple gaussians

    where `n` is the number of Gaussian components in the profile.

    Required Args:
        N/A

    Optional Args:
        peak (float): center of gaussian in pulse phase, default: ``0.5``
        width (float): stdev of pulse in pulse phase, default: ``0.1``
        amp (float): amplitude of pulse relative to other pulses, default: ``1``

    Pulses are normalized so that maximum is 1.
    See the `Pulsar._make_amp_pulses()`, `Pulsar._make_pow_pulses()`, and
    `Pulsar.make_pulses()` methods for more details.
    """
    def __init__(self, peak=0.5, width=0.05, amp=1):
        #TODO: error checking for array length consistency?
        #TODO: if any param is a not array, then broadcast to all entries of other arrays?

        super().__init__(peak=peak, width=width, amp=amp)


    def set_Nchan(self, Nchan):
        """
        Method to reintialize the portraits with the correct number of frequency
        channels. Once must run `init_profiles` or `calc_profiles` to remake
        the `profiles` property. 
        
        Note - No phases attribute, function must be updated.

        Parameters
        ----------

        Nchan : int
            Number of frequency channels.
        """
        raise NotImplementedError()
        #self.__init__(self._profiles[0], phases=self._phases, Nchan=Nchan)


class UserProfile(PulseProfile):
    """
    User specified pulse profile

    :class:`UserProfile`\'s are specified by a function used to compute the
    profile at arbitrary pulse phase. If you want to generate a profile
    from empirical data, i.e. a Numpy array, use :class:`DataProfile`.

    Required Args:

        profile_func (callable): a callable function to generate the profile
            as a function of pulse phase. This function takes a single,
            array-like input, a phase or list of phases.

    Profile is renormalized so that maximum is 1.
    See the `Pulsar._make_amp_pulses()`, `Pulsar._make_pow_pulses()`, and
    `Pulsar.make_pulses()` methods for more details.
    """
    def __init__(self, profile_func):
        # _generator is not a property, it has no setter or getter
        self._generator = profile_func

    def calc_profile(self, phases):
        """
        Calculate the profile at specified phase(s)

        Args:
            phases (array-like): phases to calc profile
        Note:
            The normalization can be wrong, if you have not run
            ``init_profile`` AND you are generating less than one
            rotation.
        """
        self._profile = self._generator(phases)
        self._Amax = self.Amax if hasattr(self, '_Amax') else np.max(self.profile)
        return self.profile / self.Amax

class DataProfile(DataPortrait):
    """
    A set of pulse profiles generated from data.

    The data are samples of the profile at specified phases. If you have a
    functional form for the profile use :class:`UserProfile` instead.

    Required Args:
        profile (array-like): profile data

    Optional Args:
        phases (array-like): list of sampled phases. If phases are omitted
            profile is assumed to be evenly sampled and cover one whole
            rotation period.

    Profile is renormalized so that maximum is 1.
    See the `Pulsar._make_amp_pulses()`, `Pulsar._make_pow_pulses()`, and
    `Pulsar.make_pulses()` methods for more details.
    """
    def __init__(self, profiles, phases=None, Nchan=None):
        # Check that no profile bins are below zero intensity
        if np.any(profiles < 0.0):
            log.warning("Some phase bins of input profile are negative, replacing them with zeros...")
            neg_idxs = np.where(profiles < 0.0)[0]
            profiles[neg_idxs] = 0.0
            
        self._phases = phases
        if profiles.ndim == 1:
            if Nchan is None:
                Nchan = 1

            profiles = np.tile(profiles,(Nchan,1))

        super().__init__(profiles=profiles, phases=phases)

    def set_Nchan(self, Nchan):
        """
        Method to reintialize the portraits with the correct number of frequency
        channels. Once must run `init_profiles` or `calc_profiles` to remake
        the `profiles` property.
        
        Note - Has same issue as set_Nchan before.

        Parameters
        ----------

        Nchan : int
            Number of frequency channels.
        """
        raise NotImplementedError()
        #self.__init__(self._profiles[0], phases=self._phases, Nchan=Nchan)
