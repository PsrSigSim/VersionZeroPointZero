"""burst.py
module to create single pulses for FRB type searches
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.interpolate as interp
from scipy import stats
from . import PSS_utils as utils


class Burst(object):
    def __init__(self, Signal_in, burst_width = 200, amplitude=100, DM_broadening=True): #period in milliseconds
        """burst_width given in microseconds"""
        self.Signal_in = Signal_in
        self.signal = self.Signal_in.signal
        self.f0 = self.Signal_in.f0
        self.bw = self.Signal_in.bw
        self.Nf = self.Signal_in.Nf
        self.Nt = self.Signal_in.Nt
        self.TotTime = self.Signal_in.TotTime
        self.SignalType = self.Signal_in.SignalType
        self.TimeBinSize = self.TotTime/self.Nt
        self.Time_location = int(Signal_in.Nt//2)
        self.burst_width_time = burst_width # In microseconds
        self.burst_width_bins = int(self.burst_width_time // self.TimeBinSize)
        self.phase = np.linspace(0., 10*self.burst_width_bins, 10*self.burst_width_bins)
        self.gamma_shape = 1
        self.gamma_scale = 2
        self.gauss_draw_sigma = 1
        self.BurstDict = {}
        self.gauss_template(peak=5*self.burst_width_bins, width=self.burst_width_bins, amp=amplitude)
        #1./np.sqrt(2.*np.pi)/self.burst_width_bins * np.exp(-0.5 * ((self.phase-0.25)/self.burst_width_bins)**2)


    def draw_intensity_pulse(self, reps):
        """draw_intensity_pulse(pulse)
        draw a single pulse as bin by bin random process (gamma distr) from input template
        shape and scale parameters can be set when Pulsar class intialized
        """
        pr = np.tile(self.profile, reps)
        L = len(pr)
        pulse = pr * self.gamma_draw_norm * np.random.gamma(self.gamma_shape, scale=self.gamma_scale, size=L)
        #pull from gamma distribution

        return pulse

    def draw_voltage_pulse(self, reps):
        """draw_voltage_pulse(pulse)
        draw a single pulse as bin by bin random process (normal distr) from input template
        """
        pr = np.tile(self.profile, reps)
        L = len(pr)
        pulse = pr * self.gauss_draw_norm * np.random.normal(0, self.gauss_draw_sigma, L) #pull from gaussian distribution

        return pulse


    #First profiles
    def gauss_template(self, peak=0.25, width=0.05, amp=1.):

        #TODO: error checking for array length consistency?
        #TODO: if any param is a not array, then broadcast to all entries of other arrays?

        try: # is this an array
            peak = np.array(peak)
            width = np.array(width)
            amp = np.array(amp)
            self.BurstDict["amplitude"] = amp
            self.BurstDict["Profile"] = "multiple gaussians"
            #amp = amp/amp.sum()  # normalize sum
            profile = np.zeros(self.burst_width_bins)
            for ii in range(amp.size):
                norm = amp[ii]/np.sqrt(2.*np.pi)/width[ii]
                self.profile += norm * np.exp(-0.5 * ((self.phase-peak[ii])/width[ii])**2)
        except: # one gaussian
            #norm = 1./np.sqrt(2.*np.pi)/width norm *
            self.profile =  np.exp(-0.5 * ((self.phase-peak)/width)**2)
            self.BurstDict["amplitude"] = amp
            self.BurstDict["Profile"] = "gaussian"

        self.BurstDict["peak"] = peak
        self.BurstDict["width"] = width

    def user_template(self,template):
        # Function to make any given 1-dimensional numpy array into the profile
        #TODO Allow other input files
        #TODO Adds error messages if not the correct type of file.
        self.BurstDict["Profile"] = "user_defined"
        #TODO Add I/O for adding attributes for user defined templates.
        self.BurstDict["peak"] = "None"
        self.BurstDict["width"] = "None"
        self.BurstDict["amplitude"] = "None"
        #self.nBinsPeriod = len(template)
        #self.profile = template
        self.nBinsTemplate = len(template)

        if self.nBinsTemplate==self.burst_width_bins:
            self.profile = template

        elif self.nBinsTemplate > self.burst_width_bins:
            self.profile = utils.rebin(template, self.burst_width_bins)
            print("User supplied template has been downsampled.")
            print("Input array length= ", self.nBinsTemplate,". Pulse template length= ",self.profile.size,".")

        else:
            TempPhase = np.linspace(0,1,len(template))
            ProfileFcn = interp.interp1d(TempPhase, template, kind='cubic', bounds_error=True)
            self.profile = ProfileFcn(self.phase)
            print("User supplied template has been interpolated using a cubic spline.")
            print("Input array length was ", self.nBinsTemplate," bins. New pulse template length is ",self.profile.size,".")

        self.MinCheck = np.amin(self.profile)
        if self.MinCheck < 0 :
            self.profile = np.where(self.profile > 0, self.profile, self.profile-self.MinCheck)

    def make_burst(self, SignalType = "intensity"):
        #Function that makes a pulse using the defined profile template
        pulseType = {"intensity":"draw_intensity_pulse", "voltage":"draw_voltage_pulse"}
        pulseTypeMethod = getattr(self, pulseType[SignalType])

        if self.SignalType == 'voltage':
            self.profile = np.sqrt(self.profile)/np.sqrt(np.amax(self.profile)) # Corrects intensity pulse to voltage profile.
            NRows = 4
            gauss_limit = stats.norm.ppf(0.999, scale=self.gauss_draw_sigma)
            # Sets the limit so there is only a small amount of clipping because of dtype.
            self.gauss_draw_norm = self.Signal_in.MetaData.gauss_draw_max/gauss_limit
            # Normalizes the 99.9 percentile to the dtype maximum.
        else:
            gamma_limit=stats.gamma.ppf(0.999,self.gamma_shape,scale=self.gamma_scale)
            # Sets the limit so there is only a small amount of clipping because of dtype.
            self.gamma_draw_norm = self.Signal_in.MetaData.gamma_draw_max/gamma_limit
            # Normalizes the 99.9 percentile to the dtype maximum.

        self.signal[:,self.Time_location:self.Time_location+len(self.profile)] += np.tile(pulseTypeMethod(1),(self.Nf,1)).astype(self.Signal_in.data_type)
        #if self.DM_broadening==True:
        #    profile_table = np.zeros((self.Nf, self.burst_width_bins))
        #    self.signal[ii,self.Time_location:self.Time_location+len(self.profile)] = np.tile(pulseTypeMethod(1),(self.Nf,1))
        #else:
        #    self.signal[:,self.Time_location:self.Time_location+len(self.profile)] = np.tile(pulseTypeMethod(1),(self.Nf,1))

        self.BurstDict["SignalType"] = SignalType
        self.BurstDict['profile'] = self.profile
        self.Signal_in.MetaData.AddInfo(self.BurstDict)
