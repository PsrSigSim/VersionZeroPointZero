from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import math as m
import pylab as pylab
import scipy as sp
from scipy import stats
from scipy import constants
from scipy.optimize import curve_fit
from scipy import misc
import scipy.ndimage as  ndimage
from . import PSS_utils as utils
from . import PSS_plot
try:
    import pyfftw
except:
    pass

KOLMOGOROV_BETA = 11.0/3
c = 2.998e8 #m/s
#lam = c/nu
KPC_TO_CM = 3.086e21
KPC_TO_M = 3.086e19
RAD_TO_MAS = 2.063e8
r_e = 2.8179403227e-15 #m


class phase_screen:
    def __init__(self, signal_object, scint_param_model='SC', Freq_DISS=None, DM='use_ism', Nx=200, Ny=200, \
                Number_r_F = 1./64, spectral_ind=KOLMOGOROV_BETA, \
                D_pulsar=1, D_screen=0.5, inner=None, outer=None, \
                rmsfres=None, apply_inner=False,  apply_outer=False):
        """
        Generates Nx x Ny points of a realization of power-law noise with unit
        variance with spectral index, spectral_ind and inner and outer scales as
        specified for a sample interval dx.

        C_n^2, the power-law amplitude of electron-density wavenumber spectrum, is calculated
        seperately by the cnsq_calc() method.

        input:
    	spectral_ind = spectral index of power-law wavenumber spectrum
    	rmsfres = rms phase at Fresnel scale (rad)
            length scales: all dimensionless:
    		freq	= frequency in MHz
    		inner, outer	= inner and outer scales
    		dx, dy		= sample intervals
    		xwidth, ywidth	= screen extent
            D = distance to screen in kPc
            D_p = Distance to Pulsar in kPc
            scint_bandwidth = scintillation bandwidth in MHz
            scint_param_scaling_model = 'SC' , 'Bhat'
                (SC) Stinebring and Condon, 1990
                (Bhat) Bhat, et al.

    	logical:
            apply_inner
            apply_outer
            normfres 	= True implies normalization to rmsfres

            Definition of Fresnel scale: r_F^2 = \lambda D / 2\pi


        returns:
           xvec, yvec, xseries, xseries_norm, qxvec, qyvec, qshape
           (xvec, yvec) coordinates perpendicular to line of sight
           xseries = screen phase
           xseries_norm = screen phase scaled to input rms phase on Fresnel scale
           qxvec, qyvec  = wavenumber axes
           qshape = sqrt(shape of wavenumber spectrum)

        references: CR98 is Cordes and Rickett, 1998
                    LK05 is Lorimer and Kramer, 2005
        """
        self.PhaseScreen_Dict={}

        if scint_param_model == 'Bhat':
            if DM == 'use_ism':
                DM = signal_object.MetaData.DM
            scat_timescale = 10**(-6.46+0.154*np.log10(DM)+1.07*(np.log10(DM))**2-3.86*np.log10(signal_object.f0/1e3))
            self.Freq_diss = 0.957/(2*np.pi*scat_timescale*1e-3)/1e6
            self.PhaseScreen_Dict['PhScreen_DM'] = DM
        elif scint_param_model == 'SC':
            self.Freq_diss = Freq_DISS

        max_freq = signal_object.freq_Array.max()
        min_freq = signal_object.freq_Array.min()
        self.r_Fresnel = np.sqrt(c / (max_freq * 1e6) * (D_screen * KPC_TO_M) / (2.0*np.pi))

        self.PhaseScreen_Dict['min_Fresnel_radius'] = self.r_Fresnel

        #Use the uncertainty relation from CR98 (C_1=1.16) to calculate the length of the hypotenuse for scattering

        #TODO Could Use the scat_timescale calculated at each frequency...
        #self.PhaseScreen_Dict['scat_timescale_f0'] = scat_timescale
        self.PhaseScreen_Dict['DISS_decorr_bw_f0'] = self.Freq_diss

        #Below we use ray optics to find the Field Coherence Scale, though other approaches are possible
        #delta_x = c * scat_timescale * 1e-3 # extra distance from scattering timescale
        #self.s_0 = np.sqrt(2 * delta_x * D_screen + delta_x**2) #Field Coherence Scale in meters
        #factor = self.s_0 / self.r_Fresnel # multiplicative factor since the uncertainty is based on 1/e time.

        xwidth = Number_r_F * self.r_Fresnel * Nx
        ywidth = Number_r_F * self.r_Fresnel * Ny
        xmax = xwidth/2
        ymax = ywidth/2
        self.PhaseScreen_Dict['PhScreen_Nx'] = Nx
        self.PhaseScreen_Dict['PhScreen_Ny'] = Ny
        self.PhaseScreen_Dict['PhScreen_xwidth'] = xwidth
        self.PhaseScreen_Dict['PhScreen_ywidth'] = ywidth

        #print("Screen Dimensions = ", round(xwidth,1) ,' x ', round(ywidth,1) , ' meters')
        print('Central Frequency decorrelation Bandwidth = ',round(self.Freq_diss,3),' MHz')
        #print('Central Freq scattering timescale = {:.2e}'.format(float(scat_timescale)),' milliseconds')
        #print( scat_timescale)

        #print('Field Coherence Scale', s_0)
        #print('Fresnel Length:', self.r_Fresnel)
        #print('xwidth = ', xwidth)
        #print('s0/r_F', factor)

        #lscale_index = 2. / (si_kol_d-2.)
        #u = max(1., rmsfres_d**lscale_index)
        #ld = rfres / u
        #lr = rfres * u
        #inners = 2.*ld

        #Nx = 30#int(xwidth//dx) #Number of x bins
        #Ny = 30#int(ywidth//dy) #Number of y bins

        self.dx = xwidth / Nx
        self.dy = ywidth / Ny
        #########################################

        self.xvec = (np.arange(0.,Nx)- Nx//2 + 1)*self.dx # Make arrays that are centered on zero, over the given widths
        self.yvec = (np.arange(0.,Ny)- Ny//2 + 1)*self.dy

        dqx = 2.*np.pi / xwidth #Smallest wavenumber in the given direction
        dqy = 2.*np.pi / ywidth
        qmaxx = (2.* np.pi) / (2.* self.dx) #Largest wavenumber in the given direction
        qmaxy = (2.* np.pi) / (2.* self.dy)
        Nqx = 2*int(qmaxx//dqx) # Number of wavenumber samples
        Nqy = 2*int(qmaxy//dqy)
        if Nqx != Nx:
            Nqx = Nx
        if Nqy != Ny:
            Nqy = Ny

        qxvec_centered = (np.arange(0.,Nqx)-Nqx//2+1)*dqx # Make arrays that are centered on zero, over the given widths
        self.qxvec = np.roll(qxvec_centered,Nqx//2+1) #Shift array by enough to make 0 first
        qyvec_centered = (np.arange(0.,Nqy)-Nqy//2+1)*dqy
        self.qyvec = np.roll(qyvec_centered, Nqy//2+1)

        #Set inner and outer scale factors if not otherwise set.
        if inner == None:
            inner = self.dx

        if outer == None:
            outer = xwidth*(1.2)

        qin = 2. * np.pi / inner #Set inner and outer scale wavenumber
        qout = 2. * np.pi / outer

        # 2016 Jan 1: put in upper wavenumber cutoff at array size
        # to avoid aliasing
        qmax = self.qxvec.max()/2.

        qxy = np.meshgrid(self.qxvec, self.qyvec, indexing='ij')

        qsq = qxy[0]**2 + qxy[1]**2
        self.qshape = (qout**2 + qsq)**(-spectral_ind/4.) * np.exp(-qsq/(2.*qmax**2))

        if apply_outer:
            self.qshape *= np.exp(-qout**2 / (2.*qsq))
            print('Applying outer-scale rolloff.')

        #The following is for use by the images() class to make the FFT(Fresnel kernel).
        qxy_centered = np.meshgrid(qxvec_centered, qyvec_centered, indexing='ij')
        self.qsq_centered =  qxy_centered[0]**2 + qxy_centered[1]**2

        ## new 2016 Jan 1:  create real white noise in x domain and FFT
        ## to get Hermitian noise
        rand_pull_r = np.random.randn(Nqx, Nqy)
        rand_pull_i = np.random.randn(Nqx, Nqy)

        #LK05 Version, From Cordes, et al. 1990
        def CnSq_calc(f,Freq_DISS,d=0.5):
            return 0.002*(f/1e3)**(3.67)*(d)**(-1.83)*(Freq_DISS)**(-0.83)

        xformr = rand_pull_r * self.qshape
        xformi = rand_pull_i * self.qshape
        xform = xformr + 1j*xformi
        self.phi = np.real(np.fft.ifft2(xform))
        self.phi /= (self.dx*self.dy)*(Nx*Ny)
        self.phi_norm = []
        for ii, freq in enumerate(signal_object.freq_Array):
            #scat_timescale = 10**(-6.46 + 0.154*np.log10(DM) + 1.07*(np.log10(DM))**2 - 3.86*np.log10(freq/1e3))
            #Freq_diss = 0.957 / (2*np.pi*scat_timescale*1e-3) / 1e6 # C1=1.16 for a uniform medium
            if scint_param_model == 'Bhat':
                Freq_diss = Bhat_Scint_Param(DM,freq)
            elif scint_param_model == 'SC':
                Freq_diss = scale_dnu_d(self.Freq_diss,signal_object.f0,freq)

            wave_num_to_phi = np.sqrt((2*np.pi)**3 * (freq*1e6/c)**2 \
                                        * 0.0198339 * (D_screen * KPC_TO_M))#0.0330054 * D_screen)#

            self.phi_norm = np.append(self.phi_norm, \
                                    np.sqrt(CnSq_calc(freq,Freq_diss)) * wave_num_to_phi)
            #Using Michael Lam's version, Cordes and Rickett, 1998
            #self.phi_norm = np.append(self.phi_norm, \
            #                        np.sqrt(cnsq_calc(taud=scat_timescale*1e-3, nu=freq)) * wave_num_to_phi)

        self.Nx = Nx
        self.Ny = Ny
        self.xmax = xmax
        self.ymax = ymax
        self.Nf = signal_object.Nf
        self.freq_Array = signal_object.freq_Array

        # Normalization factor needs to be calculated on pure power-law spectrum
        # before any rolloff at the refraction scale
        if rmsfres!= None: #This normalization gives the desired DM_rms desired at the end.
            self.phi_normfres = np.zeros(self.phi.shape)
            frindx = int(self.r_Fresnel//self.dx)
            x1dcut = self.phi[0,:]
            var_fres_in = np.var(x1dcut[0:np.size(x1dcut)-frindx]-x1dcut[frindx:])
            norm_factor = rmsfres / np.sqrt(var_fres_in)
            self.phi_normfres = self.phi * norm_factor
            xn1dcut = self.phi_normfres[0,:]
            var_fres_out = np.var(xn1dcut[0:np.size(xn1dcut)-frindx]-xn1dcut[frindx:])

            print("Index of Fresnel scale = ", frindx)
            print('Phas RMS In:',var_fres_in,' Phase RMS Out:', var_fres_out)



        # applying inner scale now an option with apply_inner = True
        # needs to be applied *after* normalization!
        # now need to recalculate the realization and apply norm_factor
        if apply_inner:
            print("Applying inner-scale rolloff")
            self.qshape_rolloff = np.exp(-qsq / (2.*qin**2))

           # Recalculate
            xform *= self.qshape_rolloff
            self.spectrum = abs(xform)**2
            self.phi = np.real(np.fft.ifft2(xform))

            if rmsfres!= None:
                self.phi_normfres = self.phi * norm_factor


class images(object):
    def __init__(self, phase_screen, signal_object, fourier_mode=True, speed='slow', mode='explore'):

        if phase_screen.phi.ndim == 3:
            phase_norm = np.ones(signal_object.Nf)
        elif phase_screen.phi.ndim == 2:
            phase_norm = phase_screen.phi_norm

        xy = np.meshgrid(phase_screen.xvec, phase_screen.yvec, indexing='ij')
        rsqvec = xy[0]**2 + xy[1]**2

        if mode == 'explore':
            self.gain = np.zeros((phase_screen.Nf,phase_screen.Nx,phase_screen.Ny))
            self.kernel = np.zeros(self.gain.shape, dtype='complex64')
            self.kernelFFT = np.zeros(self.gain.shape, dtype='complex64')
            self.screenFFT = np.zeros(self.gain.shape, dtype='complex64')
            self.field = np.zeros(self.gain.shape, dtype='complex64')
        elif mode == 'simulation':
            self.gain = np.zeros((phase_screen.Nf,phase_screen.Nx))

        for ii, freq in enumerate(phase_screen.freq_Array):
            r_Fres_squared = r_Fres_SQ(freq)
            #SD = phase_screen.xmax*phase_screen.ymax#np.exp(-rsqvec / SD)*
            if fourier_mode:
                FresnelKernel_Norm = 1 #np.sqrt(r_Fres_squared*np.pi)/2
                kernel0fft = FresnelKernel_Norm * (1+1j) \
                                * np.exp(-1j*phase_screen.qsq_centered/2 \
                                *r_Fres_squared)
                screen0 = np.exp(phase_norm[ii] * 1j * phase_screen.phi) #[ii,:,:])
                if speed == 'slow':
                    screen0fft = np.fft.fft2(screen0)
                if speed == 'fast':
                    screen0fft = pyfftw.interfaces.scipy_fftpack.fft2(screen0)
                if speed == 'fastest':
                    screen0fft = 2

                field0fft = kernel0fft * screen0fft
                if mode == 'explore':
                    self.kernelFFT[ii,:,:] = kernel0fft
                    self.screenFFT[ii,:,:] = screen0fft

            else:
                kernel0 = np.exp(1j * rsqvec / (2. * r_Fres_squared)) # See Eq (2.1), (Narayan, 1992)
                screen0 = np.exp(phase_norm * 1j * phase_screen.phi[ii])
                kernel0fft = np.fft.fft2(kernel0)
                screen0fft= np.fft.fft2(screen0)
                field0fft = kernel0fft * screen0fft
                if mode == 'explore':
                    self.kernel[ii,:,:] = kernel0

            norm = 1/np.sqrt(2)#((phase_screen.dx*phase_screen.dy)/( r_Fres_squared))/(phase_screen.Nx*phase_screen.Ny)#2.*np.pi*
            if speed == 'slow':
                field = norm * np.fft.fftshift(np.fft.ifft2(field0fft))
            if speed == 'fast':
                field = norm * pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.scipy_fftpack.ifft2(screen0))
                #probably should del field and just straight calculate the gain?

            if mode == 'explore':
                self.field[ii,:,:] = field
                self.gain[ii,:,:] = abs(field**2)
            elif mode == 'simulation':
                self.gain[ii,:] = abs(field**2)[:,int(phase_screen.Ny//2)]
            signal_object.MetaData.AddInfo(phase_screen.PhaseScreen_Dict)

    ###################### Plots ########################

    def dynamic_spectrum(self, signal_object, **kwargs):
        return PSS_plot.dynamic_spectrum(self, signal_object, **kwargs)

    def gain_pdf(self, **kwargs):
        return PSS_plot.gain_pdf(self, **kwargs)


def r_Fres_SQ(freq, wavelength=None, D=0.5, units=['MHz','kPc']):
    """Returns Fresnel length scale. In desired units"""
    if units != ['MHz','kPc']:
        raise ValueError(units, ': units not currently supported!')

    MHz = 1e6
    D *= KPC_TO_M
    if wavelength == None:
        wavelength = c/(freq*MHz)

    return wavelength * D / (2*np.pi)

def Bhat_Scint_Param(DM,frequency, Output='DeltaF_diss'):
    """
    Calculate the dependence of scintillation parameters versus DM and frequency.
    """
    scat_timescale = 10**(-6.46+0.154*np.log10(DM)+1.07*(np.log10(DM))**2-3.86*np.log10(frequency/1e3))
    if Output=='DeltaF_diss':
        Out = 0.957/(2*np.pi*scat_timescale*1e-3)/1e6 #Calculate Frequency Decorrelation bandwidth
    elif Output=='Scat_time':
        Out = scat_timescale
    return Out

def cnsq_calc(nu=1000, dnud=None, taud=None, dtd=None, D=1, PM=None, beta=11./3.,mode='screen',ds=0.001):
    '''
    Function for calculating the electron-density wavenumber spectrum coefficient.
    Following Cordes and Rickett 1998 and subsequent references

    Inputs:
    nu = frequency in MHz
    dtd = \Delta t_d, the scintillation timescale in seconds
    dnud = \Delta \nu_d, the scintillation bandwidth in Hz
    taud = \tau_d, the scattering timescale in seconds
    D = distance in kpc
    beta = power-law index of electron-density wavenumber spectrum
    mode = screen, uniform

    Outputs:
    cnsq = C_n^2, the power-law amplitude of electron-density wavenumber spectrum
    '''

    if mode == 'screen' and beta == KOLMOGOROV_BETA:
        C1 = 0.957
    elif mode == 'screen' and beta == 4:
        C1 = 1.0
    elif mode == 'uniform' and beta == KOLMOGOROV_BETA:
        C1 = 1.16
    elif mode == 'uniform' and beta == 4:
        C1 = 1.53
    else:
        C1 = 1.0 ### See Lambert and Rickett 1998 for calculation

    if taud is None and dnud is None:
        raise ValueError("Require either taud or dnud to be set.")
    elif taud is None:
        taud = C1 / (2*np.pi*dnud)
    elif dnud is None:
        dnud = C1 / (2*np.pi*taud)

    if PM is None:
        vp = 100*1e3 #m/s
    else:
        vp = 4.74*1e3*PM #m/s


    if mode == 'screen': ### For now assume screen distance is half way
        if taud is not None:
            eta0ds = 2*c*taud / (0.5*D*KPC_TO_M*(1-0.5))
        elif dtd is not None:
            pass
        eta0 = eta0ds / (ds*KPC_TO_M)
    elif mode == 'uniform':
        eta0 = 0

    nu *= 1e6
    eta0 = eta0 * KPC_TO_M * RAD_TO_MAS**2
    #print(eta0)
    #TODO Add an inner scale parameter to this method
    l1 = 100000 #m
    #cnsq = (eta0 / 3.2) * (nu/1e9)**4 * (l1/100000)**(1.0/3) * 10**-3.5
    cnsq = 0.002*(nu/1e9)**4 * (D)**(-1.83) * (dnud/1e6)**(-0.83)
    #print("C_n_squared:",cnsq, np.log10(cnsq))

    return cnsq

'''
Written by Michael Lam, 2017
Scale dnu_d and dt_d based on:
dnu_d propto nu^(22/5)
dt_d propto nu^(6/5) / transverse velocity
See Stinebring and Condon 1990 for scalings with beta (they call it alpha)

Be careful with float division now. This has been removed to allow for numpy arrays to be passed through.
'''

def scale_dnu_d(dnu_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = 2.0*beta/(beta-2) #(22.0/5)
    elif beta > 4:
        exp = 8.0/(6-beta)
    return dnu_d*(nu_f/nu_i)**exp

def scale_dt_d(dt_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = 2.0/(beta-2) #(6.0/5)
    elif beta > 4:
        exp = float(beta-2)/(6-beta)
    return dt_d*(nu_f/nu_i)**exp

def scale_tau_d(tau_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = -2.0*beta/(beta-2) #(-22.0/5)
    elif beta > 4:
        exp = -8.0/(6-beta)
    return tau_d*(nu_f/nu_i)**exp

def scale_dt_r(tau_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = beta/float(2-beta) #-2.2
    elif beta > 4:
        exp = 4.0/(beta-6)
    return tau_d*(nu_f/nu_i)**exp
