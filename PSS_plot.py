"""PSS_plot.py
A set of plotting commands which uses the metadata of the signal to make plotting simpler.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import matplotlib.pyplot as plt
import numpy as np
from . import PSS_utils as utils

__all__= ['profile_plot','pulse_plot','filter_bank','dynamic_spectrum','gain_pdf','plot_dispersed','joy_division_profiles']

#plt.rcParams['figure.figsize'] = (8.0,6.0)
plt.rcParams.update({'font.size': 14})

def profile_plot(signal_object, freq_bin=0, phase=False, **kwargs):
    Title = 'Pulsar Profile Template'
    try:
        profile = signal_object.MetaData.profile
    except:
        raise ValueError('Need to sample pulses!')
    Nx = profile.shape[1]

    if phase:
        Phase = np.linspace(0., 1, Nx)
        plt.plot(Phase, profile[0], lw=0.7, **kwargs)
        plt.xlabel('Phase')
        #plt.ylabel(Y_label)
        plt.yticks([])
        plt.ylim(0,profile.max()*1.05)

    else:
        stop_time = signal_object.MetaData.pulsar_period
        time = np.linspace(0, stop_time, Nx)
        plt.plot(time, profile[0], lw=0.7, **kwargs)
        plt.xlabel('Time (ms)')
        #plt.ylabel(Y_label)
        plt.ylim(0,profile.max()*1.05)
        #plt.yticks([])

    plt.title(Title)
    plt.show()
    plt.close()

def joy_division_profiles(pulsar_object, step=1, N=10,Title='Profile by Frequency Channel'):
    array=pulsar_object.profile
    try:
        array_len = len(N)
        jj = 0
        for ii, jj in enumerate(N):
            plt.plot(array[jj,:]+ii*(step),c='k')
        n=len(N)
    except:
        Freq_channels = len(array[:,0])
        Freq_step = int(Freq_channels//N)
        for ii in range(N):
            plt.plot(array[ii*Freq_step,:]+ii*(step),c='k')
        n=N
    plt.title(Title)
    plt.xlabel('Phase')
    plt.ylabel('Frequency Channel')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0,len(array[0,:]))
    plt.ylim(0,n*step+0.1)
    plt.show()
    plt.close()

def pulse_plot(signal_object, N_pulses=1, pol_bin=0, freq_bin=0, start_time=0, phase=False, **kwargs):
    """N_pulses : int, optional
            Number of pulses to plot.
        pol_bin : int, optional

        freq_bin : int, optional
            Number of frequency bins.
        start_time : float, optional
            Starting time value of pulse plot in milliseconds.
        phase : bool, optional
            'True' to plot as function of phase.
            'False' to plot as function of time (ms).

    """
    try:
        nBins_per_period = int(signal_object.MetaData.pulsar_period//signal_object.TimeBinSize)
    except:
        raise ValueError('Need to sample pulses!')

    if signal_object.SignalType == 'intensity':
        Y_label = 'Relative Intensity'
        Title = 'Pulse Intensity'
        row = freq_bin
    if signal_object.SignalType == 'voltage':
        Y_label = 'Relative Voltage'
        Title = 'Pulse Voltage'
        row = pol_bin

    if phase:
        Phase = np.linspace(0., N_pulses, N_pulses*nBins_per_period)
        plt.plot(Phase, signal_object.signal[row,:N_pulses*nBins_per_period], lw=0.4, **kwargs)
        plt.xlabel('Phase')
        #plt.ylabel(Y_label)
        #plt.yticks([])

    else:
        stop_time = start_time + N_pulses*nBins_per_period*signal_object.TimeBinSize
        Nx = N_pulses*nBins_per_period
        time = np.linspace(start_time, stop_time, Nx)
        plt.plot(time,signal_object.signal[row,:Nx], lw=0.4, **kwargs)
        plt.xlabel('Time (ms)')
        #plt.ylabel(Y_label)
        #plt.yticks([])

    plt.title(Title)
    plt.show()
    plt.close()


def filter_bank(signal_object, grid=False, N_pulses=1, start_time=0, phase=False, **kwargs):
    try:
        nBins_per_period = int(signal_object.MetaData.pulsar_period//signal_object.TimeBinSize)
    except:
        raise ValueError('Need to sample pulses')

    if signal_object.SignalType == 'intensity':
        #Y_label = 'Relative Intensity'
        Title = 'Filter Bank'
    if signal_object.SignalType == 'voltage':
        #Y_label = 'Relative Voltage'
        #Title = 'Filter Bank'
        raise ValueError('Filter Bank not supported for voltage signals at this time.')

    stop_bin = N_pulses*nBins_per_period
    Y_label = 'Frequency (MHz)'
    if phase:
        Extent = [0, N_pulses, signal_object.first_freq, signal_object.last_freq]
        plt.imshow(signal_object.signal[:,:stop_bin],origin='left',cmap='plasma',aspect='auto',extent=Extent,**kwargs)
        #ax = plt.gca();
        plt.xlabel('Phase')
        plt.ylabel(Y_label)
        #plt.yticks([])
        if grid:
            ax = plt.gca();
            ax.set_xticks([])#np.linspace(Extent[0], Extent[1], N_pulses*2));
            ax.set_yticks([])#np.linspace(Extent[2], Extent[3], 5));
            ax.set_xticks(np.linspace(Extent[0], Extent[1], N_pulses*10), minor=True);
            ax.set_yticks(np.linspace(Extent[2], Extent[3], signal_object.Nf), minor=True);
            ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)

    else:
        time = len(signal_object.signal[0,:])
        stop_time = start_time + N_pulses * nBins_per_period * signal_object.TimeBinSize
        Extent = [start_time, stop_time, signal_object.first_freq, signal_object.last_freq]
        plt.imshow(signal_object.signal[:,:stop_bin],origin='left',cmap='plasma',aspect='auto',extent=Extent,**kwargs)

        plt.xlabel('Observation Time (ms)')
        plt.ylabel(Y_label)
        #plt.yticks([])
        if grid:
            ax = plt.gca();
            ax.set_xticks(np.linspace(Extent[0], Extent[1], 20), minor=True);
            ax.set_yticks(np.linspace(Extent[2], Extent[3], signal_object.Nf), minor=True);
            ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)

    plt.title(Title)
    plt.show()
    plt.close()

def gain_pdf(image_screen, which_sample='middle'):

    if which_sample == 'middle' and image_screen.gain.ndim == 3:
        which_sample= image_screen.gain.shape[1]//2
        gain_mean = image_screen.gain[:,which_sample,:].mean()
        plt.hist(image_screen.gain[:,which_sample,:].flatten(), \
                    bins=60, normed=True, label='Sampled Gains')
    elif which_sample != 'middle' and image_screen.gain.ndim == 3:
        gain_mean = image_screen.gain[:,which_sample,:].mean()
        plt.hist(image_screen.gain[:,which_sample,:].flatten(), \
                    bins=60, normed=True, label='Sampled Gains')
    elif image_screen.gain.ndim == 2:
        gain_mean = image_screen.gain.mean()
        plt.hist(image_screen.gain.flatten(), \
                    bins=60, normed=True, label='Sampled Gains')

    g = np.linspace(0,10,100)
    plt.plot(g, np.exp(-g),label=r'Theoretical Gain, $e^{-g}$')
    plt.title('PDF of Scintillation Screen Gains \n'+ r'$\left< g \right>=$'+str(round(gain_mean,3)))
    plt.xlabel('Gain')
    plt.ylabel('Probability Density')
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.show()
    plt.close()


def dynamic_spectrum(image_screen, signal_object, save=False, window_size = 'optimal', **kwargs):
    """
    window_size = optimal or full
    """
    S = signal_object
    image = image_screen

    if image.gain.ndim == 3:
        nfreqs, Nx, Ny = image.gain.shape
        Normalized_Intensity = image.gain[:,:,Ny//2]
        Normalized_Intensity -= Normalized_Intensity.mean()
    elif image.gain.ndim == 2:
        nfreqs, Nx = image.gain.shape
        Normalized_Intensity = image.gain
        Normalized_Intensity -= Normalized_Intensity.mean()
        #Mean of gain should be 1, but will vary in different realizations.

    ACF = utils.acf2d(Normalized_Intensity, mode='same') #2d autocorrelation
    ACF /= np.amax(ACF)
    middle_freq, middle_time = np.unravel_index(np.argmax(ACF), ACF.shape)

    f, ax = plt.subplots(2, 2)
    f.set_figheight(15)
    f.set_figwidth(15)

    ##Dynamic Spectrum

    ExtentDS = [0,Nx,S.freq_Array[0],S.freq_Array[-1]]

    ax[0, 0].imshow(Normalized_Intensity, origin='left', \
                    aspect='auto', extent=ExtentDS, interpolation='none', \
                    cmap='binary')
    ax[0,0].set_ylabel('Frequency (MHz)')
    ax[0,0].get_yaxis().get_major_formatter().set_useOffset(False)
    #ax[0, 0].colorbar()

    #Autocorrelation Function Plot

    scint_bandwidth = utils.find_nearest(ACF[middle_freq:,middle_time],0.5)*S.freqBinSize
    scint_timescale = utils.find_nearest(ACF[middle_freq,middle_time:],1/np.exp(1))#*S1.freqBinSize*1e3,1)

    if window_size=='optimal':
        freq_factor = 100
        time_factor = 50
        freq_frame_size = int(scint_bandwidth//S.freqBinSize)*freq_factor
        time_frame_size = scint_timescale*time_factor
        while 2*freq_frame_size > S.Nf and freq_factor > 2:
            freq_factor -= 2
            freq_frame_size = int(scint_bandwidth//S.freqBinSize)*freq_factor
        while 2*time_frame_size > S.MetaData.PhScreen_Nx and time_factor > 2:
            time_factor -= 2
            time_frame_size = scint_timescale*time_factor

    elif window_size=='full':
        freq_frame_size = S.Nf//2
        time_frame_size = S.MetaData.PhScreen_Nx//2

    ExtentACF = [-time_frame_size,time_frame_size,\
                -freq_frame_size*S.freqBinSize,freq_frame_size*S.freqBinSize]
    ax[1, 1].imshow(ACF[middle_freq-freq_frame_size:middle_freq+freq_frame_size, \
                middle_time-time_frame_size:middle_time+time_frame_size]\
                ,aspect='auto',interpolation='none',origin='left',extent=ExtentACF,cmap='copper_r')
    ax[1, 1].set_title('Autocorrelation Function')
    ax[1, 1].set_xlabel(r'Time Lag')
    ax[1, 1].set_ylabel('Freq Lag (MHZ)')

    time_lag = np.linspace(-time_frame_size, time_frame_size, 2*time_frame_size)
    freq_lag = np.linspace(-freq_frame_size*S.freqBinSize, freq_frame_size*S.freqBinSize, \
                            2*freq_frame_size)

    ax[1, 0].plot(freq_lag, ACF[middle_freq-freq_frame_size\
                                :middle_freq+freq_frame_size, middle_time])
    ax[1, 0].plot(freq_lag, np.ones(2*freq_frame_size)*0.5,'--') #Half Maximum Cutoff
    ax[1, 0].set_title('Frequency ACF')
    ax[1, 0].set_xlim(-freq_frame_size*S.freqBinSize, freq_frame_size*S.freqBinSize)
    ax[1, 0].set_ylim(-0.05,1.05)
    #xlim([middle_freq-frame_size//2,middle_freq+frame_size//2])


    try:
        DM = S.MetaData.PhScreen_DM
    except:
        DM = 'NA'
    if round(S.MetaData.DISS_decorr_bw_f0) ==0:
        freq_units = ' kHZ'
        unit_factor = 1e3
    else:
        freq_units = ' MHZ'
        unit_factor = 1

    f.suptitle('Spacial Dynamic Spectra Check, DM= '+ str(DM)+'\n Input Scintillation BW=' + \
                str(unit_factor*round(S.MetaData.DISS_decorr_bw_f0,3)) + freq_units \
                +'\n Measured Scintillation BW=' + str(unit_factor*scint_bandwidth) + freq_units)

    ax[0, 1].plot(time_lag, ACF[middle_freq, middle_time-time_frame_size\
                                :middle_time+time_frame_size])
    ax[0, 1].plot(time_lag, np.ones(2*time_frame_size)*(1/np.exp(1)), '--') #1/e Cutoff
    ax[0, 1].set_xlim(-time_frame_size,time_frame_size)
    ax[0, 1].set_ylim(-0.05,1.05)
    ax[0, 1].set_title('Time ACF')

    f.subplots_adjust(hspace=0.3)

    if save==True:
        f.savefig('DynamicSpectrum_f0_' + str(S1.f0)+'MHz_DM_'+str(DM))
    plt.draw()
    plt.show()
    plt.close()


def plot_dispersed(signal_object, N_pulses = 2, channel = 0, **kwargs): # Plots dispersed sig produced in ISM with Disperse()
    lim = N_pulses*signal_object.MetaData.nBins_per_period
    psr_period = signal_object.MetaData.pulsar_period
    if signal_object.SignalType == 'voltage':
        plt.title('Voltage vs. Time')
        plt.ylabel('Voltage')
        plt.xlabel('Time (ms)')
        plt.xlim(0,psr_period)
    elif signal_object.SignalType == 'intensity':
        plt.title('Intensity vs. Time')
        plt.ylabel('Intensity')
        plt.xlabel('Time (ms)')
        plt.xlim(0,psr_period)
    plt.yticks([])
    max_lower = np.amax(signal_object.signal[channel,:lim])
    min_upper = np.amin(signal_object.undispersedsig[channel,:lim])
    jump = max_lower + np.abs(min_upper) + 4
    t = np.linspace(0,psr_period,lim)
    plt.plot(t, signal_object.undispersedsig[channel,:lim]+jump, c='k', **kwargs)
    plt.plot(t, signal_object.signal[channel,:lim], c='c', **kwargs)
    plt.show()
    plt.close()
    # TODO flag about if it hasn't been dispersed yet
