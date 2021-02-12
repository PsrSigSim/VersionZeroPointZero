# -*- coding: utf-8 -*-
# encoding=utf8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import fitsio
import sys
sys.path.insert(0,'/Users/hazboun/software_development/PulsarDataToolbox/')
import pdat
from .file import BaseFile
from ..utils import make_quant, make_par
from ..signal import Signal
from ..signal import FilterBankSignal
# Not sure if we can require PINT as a dependency...
import pint.models as models
import pint.polycos as polycos
import pint.toa as toa
import warnings

__all__ = ["PSRFITS"]

class PSRFITS(BaseFile):
    """A class for saving PsrSigSim signals as PSRFITS standard files.

    Parameters
    ----------

    path: str
        name and path of new psrfits file that will be saved
    obs_mode: str
        what type of observation is the data, SEARCH, PSR, etc.
    template: str
        the path and name of the template fits file that will be loaded
    copy_template: bool
        Does nothing?
    fits_mode: str
        How we want to save the data, right now just 'copy' is valid
    """

    def __init__(self, path=None, obs_mode=None, template=None,
                 copy_template=False,fits_mode='copy'):

        self._tbin = None
        self._nbin = None
        self._nsblk = None
        self._nchan = None
        self._npol = None
        self._nrows = None
        self._tsubint = None
        self._chan_bw = None
        self._obsbw = None
        self._obsfreq = None

        self._fits_mode = fits_mode

        if template is None:
            template_path = False
        else:
            template_path = template

        super().__init__(path=path)

        self.file = pdat.psrfits(psrfits_path=path, mode='rw',
                                 from_template=template_path,
                                 obs_mode=obs_mode, verbose=False)
        if obs_mode is None:
            self.obs_mode = self.file.obs_mode
        else:
            self.obs_mode = obs_mode

        #Lists of needed parameters from the PSRFITS file.
        self.pfit_pars = {'PRIMARY':['TELESCOP',
                                     'FRONTEND',
                                     'BACKEND',
                                     'OBS_MODE',
                                     'OBSFREQ',
                                     'OBSBW',
                                     'OBSNCHAN',
                                     'FD_POLN',
                                     'STT_IMJD',
                                     'STT_SMJD',
                                     'STT_OFFS'],
                          'SUBINT':['TBIN',
                                    'NAXIS',
                                    'NAXIS1',
                                    'NAXIS2',
                                    'NCHAN',
                                    'POL_TYPE',
                                    'NPOL',
                                    'NBIN',
                                    'NBITS',
                                    'CHAN_BW',
                                    'NSBLK',
                                    'DAT_SCL',
                                    'DAT_OFFS',
                                    'DAT_WTS',
                                    "TSUBINT",], # TTYPE2 is the subint length
                            'PSRPARAM':[
                                    ]
                           }

        #Might not need this, since we can calculate from other terms.
        if self.obs_mode=='SEARCH':
            self.pfit_pars['SUBINT'].append('TDIM17')
        elif self.obs_mode=='PSR':
            #self.pfit_pars['SUBINT'].append('TDIM20')
            for k in self.file.fits_template['SUBINT'].read_header().keys():
                if 'TDIM' in k:
                    self.pfit_pars['SUBINT'].append(k)
            self.pfit_pars['PSRPARAM'].append('F')
            # Need both there depending on the template file. Only one will work.
            self.pfit_pars['PSRPARAM'].append('F0')
            self.pfit_pars['PSRPARAM'].append('DM')

    # We will define a function that will generate a dictionary with the parameters needed to replace POLYCO header params
    def _gen_polyco(self, parfile, MJD_start, segLength = 60.0, ncoeff = 15, \
                       maxha=12.0, method="TEMPO", numNodes=20, usePINT = True):
        """
        This will be a convenience function to generate polycos and subsequent parameters to replace the values in a
        PSRFITS file header. The default way to do this will be to use PINT (usePINT = True), with other methods
        currently unsupported.

        Parameters
        ----------

        parfile : str
            Path to par file used to generate the polycos. The observing frequency, and observatory will
                            come from the par file
        MJD_start : float
            Start MJD of the polyco. Should start no later than the beginning of the observation
        segLength : float
            Length in minutes of the range covered by the polycos generated. Default is 60 minutes
        ncoeff : int
            Number of polyco coeffeicients to generate. Default is 15, the same as in the PSRFITS file
        maxha : float
            Max hour angle needed by PINT. Default is 12.0
        method : str
            Method PINT uses to generate the polyco. Currently only TEMPO is supported.
        numNodes : int
            Number of nodes PINT will use to fit the polycos. Must be larger than ncoeff
        usePINT : int
            Method used to generate polycos. Currently only PINT is supported.
        """
        if usePINT:
            # Define dictionary to put parameters into
            polyco_dict = {'NSPAN' : segLength, 'NCOEF' : ncoeff}
            # load parfile to PINT model object
            m = models.get_model(parfile)
            # Determine MJD_end based on segLength
            MJD_end = MJD_start + np.double(make_quant(segLength,'min').to('day').value) # MJD
            # Determine obseratory and observing frequency
            obsFreq = m.TZRFRQ.value # in MHz
            polyco_dict['REF_FREQ'] = obsFreq
            obs = m.TZRSITE.value # string
            polyco_dict['NSITE'] = obs.encode('utf-8') # observatory code needs to be in binary
            # Get pulsar frequency
            polyco_dict['REF_F0'] = m.F0.value
            # get the polycos
            pcs = polycos.Polycos()
            pcs.generate_polycos(m, MJD_start, MJD_end, obs, segLength, ncoeff, obsFreq, maxha=12.0, method="TEMPO", \
                                     numNodes=20)
            coeffs = pcs.polycoTable['entry'][-1].coeffs
            polyco_dict['COEFF'] = coeffs
            # Now we need to determine the reference MJD, and phase
            REF_MJD = np.double(pcs.polycoTable['tmid'][-1])
            polyco_dict['REF_MJD'] = REF_MJD
            # Now find the phase difference
            tmid_toa = toa.get_TOAs_list([toa.TOA(REF_MJD, obs=obs, freq=obsFreq)])
            ref_phase = m.phase(tmid_toa)
            # Need to force positive value
            if ref_phase.frac.value[0] < 0.0:
                ref_frac_phase = 1.0 - abs(ref_phase.frac.value[0])
            else:
                ref_frac_phase = ref_phase.frac.value[0]
            polyco_dict['REF_PHS'] = ref_frac_phase

            return polyco_dict

        else:
            #print("Only PINT is currently supported for generating polycos")
            raise NotImplementedError("Only PINT is currently supported for generating polycos")

    # Define a function to collect the metadata necessary for phase connection
    def _gen_metadata(self, signal, pulsar, ref_MJD = 56000.0, inc_len = 0.0):
        """
        Function for determining the remaining numbers necessary to phase connect the TOAs.
        In particular OFFS_SUB values in the subint header and STT_IMJD/SMJD/OFFS values for
        files we desire to have some phase connection.

        Parameters
        ----------

        signal : `psrsigsim.signal.Signal`
            Signal class object that will contain necessary meta-data, e.g.
            nsub, sublen, etc.
        pulsar :  `psrsigsim.signal.Pulsar`
            Pulsar class object, will contain necessary meta-data, e.g. period
        ref_MJD : float
            Initial time to reference the observations to (MJD). This value
            should be the start MJD (fraction if necessary) of the first file,
            default is 56000.0
        inc_len : float
            Time difference (days) between reference MJD and new phase connected
            MJD, default is 0 (e.g. no time difference)
        """
        # Assign appropriate units
        inc_len = make_quant(np.double(inc_len), 'day')
        init_MJD = make_quant(ref_MJD, 'day')
        # Define the dictionaries
        subint_dict = {'EPOCHS' : 'MIDTIME'} # I think that's what we want
        primary_dict = {}

        # Define the offs_sub values based on nsub and sublen
        OFF_SUBS = np.zeros(signal.nsub)
        for ii in range(signal.nsub):
            OFF_SUBS[ii] = np.double(signal.sublen.value/2.0 + ii*signal.sublen.value)
        subint_dict['OFFS_SUB'] = OFF_SUBS
        # if the increment length is 0, then all other values can be zero as well

        init_fracMJD = make_quant(np.double('0.'+str(init_MJD.value).split('.')[-1]),'day').to('s')
        init_SMJD = np.double(str(init_fracMJD.value).split('.')[0])
        init_OFFS = np.double('0.'+str(init_fracMJD.value).split('.')[-1])

        if inc_len.value == 0.0:
            next_MJD = init_MJD
            next_seconds = make_quant(init_SMJD, 's')
            next_frac_sec = make_quant(init_OFFS, 's')
            #next_frac_sec = make_quant(0.0, 's')
        else:
            # Now determine the next appropriate staring MJD, SMJD, and SOFFS
            next_MJD = init_MJD+np.floor(inc_len)
            # Now get the number of seconds (and fractional seconds) leftover
            leftover_s = (inc_len-np.floor(inc_len)).to('s')
            next_seconds = make_quant(init_SMJD, 's') + np.floor(leftover_s)
            next_frac_sec = make_quant(init_OFFS, 's') + (leftover_s - np.floor(leftover_s))

        # Assign these to dictionary values
        primary_dict['OBSFREQ'] = self.obsfreq.value
        primary_dict['OBSBW'] = self.obsbw.value
        primary_dict['CHAN_DM'] = signal.dm.value
        primary_dict['STT_IMJD'] = int(next_MJD.value)
        primary_dict['STT_SMJD'] = int(next_seconds.value)
        primary_dict['STT_OFFS'] = np.double(next_frac_sec.value)
        primary_dict['BE_DELAY'] = 0.0

        return primary_dict, subint_dict

    def _edit_psrfits_header(self, polyco_dict, subint_dict, primary_dict):
        """
        This function is used as a convienience function to edit the header
        data of a PSRFITS file, particularly the POLYCO, PRIMARY, and SUBINT
        headers.

        Parameters
        ----------

        polyco_dict : dict
            Dictionary of polyco header parameters to be replaced as generated
            by _gen_polycos() function.
        subint_dict : dict
            Dictionary of subint header parameters to be replaced as generated
            by _gen_metadata() function.
        primary_dict : dict
            Dictionary of primary header parameters to be replaced as generated
            by _gen_metadata() function.
        """
        # We go through each dictionary and replace the appropriate values; start with primary
        self.file.set_draft_header('PRIMARY', primary_dict)
        # Now we adjust the history header
        self.file.HDU_drafts['HISTORY'][0]['POL_TYPE'] = str.encode(subint_dict['POL_TYPE'])
        self.file.HDU_drafts['HISTORY'][0]['NSUB'] = self.nsubint
        self.file.HDU_drafts['HISTORY'][0]['NPOL'] = self.npol
        self.file.HDU_drafts['HISTORY'][0]['NBIN'] = subint_dict['NBIN']
        self.file.HDU_drafts['HISTORY'][0]['NBIN_PRD'] = subint_dict['NBIN']
        self.file.HDU_drafts['HISTORY'][0]['TBIN'] = subint_dict['TBIN']
        self.file.HDU_drafts['HISTORY'][0]['CTR_FREQ'] =self.obsfreq.value
        self.file.HDU_drafts['HISTORY'][0]['NCHAN'] = self.nchan
        self.file.HDU_drafts['HISTORY'][0]['CHAN_BW'] = subint_dict['CHAN_BW']
        self.file.HDU_drafts['HISTORY'][0]['DM'] = subint_dict['DM']
        # Now do the subint header
        self.file.set_draft_header('SUBINT', {'EPOCHS' : subint_dict['EPOCHS'], 'CHAN_BW' : subint_dict['CHAN_BW'],
                                              'POL_TYPE': subint_dict['POL_TYPE'], 'TBIN': subint_dict['TBIN'],
                                              'DM' : subint_dict['DM'], 'NBIN' : subint_dict['NBIN']})
        # Now replace the values of the offs_sub subints
        for ii in range(len(subint_dict['OFFS_SUB'])):
            self.file.HDU_drafts['SUBINT'][ii]['OFFS_SUB'] = subint_dict['OFFS_SUB'][ii]
            self.file.HDU_drafts['SUBINT'][ii]['TSUBINT'] = subint_dict['TSUBINT'][ii]
        # And finally the polycos;
        for ky in polyco_dict.keys():
            try:
                self.file.HDU_drafts['POLYCO'][0][ky] = polyco_dict[ky]
            except:
                print(ky)
        # And we want to get rid of many of the PSRPARAM lines, just in case?
        # NOTE: THIS IS HARDCODED AND WILL NEED TO BE FIXED EVENTUALLY
        delete_params = ["BINARY", "A1", "E", "T0", "PB", "OM", "SINI", "M2", "F1", \
                         "PMDEC", "PMRA", "TZRMJD", "TZRFRQ", "TZRSITE"]
        for param in self.file.HDU_drafts['PSRPARAM']:
            for dp in delete_params:
                if dp.encode('utf-8') == param[0].split()[0]:
                    idx = np.where(param == self.file.HDU_drafts['PSRPARAM'])[0]
                    self.file.HDU_drafts['PSRPARAM'] = np.delete(self.file.HDU_drafts['PSRPARAM'], idx)

    # Save the signal
    def save(self, signal, pulsar, parfile = None, \
             MJD_start = 56000.0, segLength = 60.0, inc_len = 0.0, \
             ref_MJD = 56000.0, usePint = True, eq_wts = True):
        """Save PSS signal file to disk. Currently only one mode of doing this
        is supported. Saved data can be phase connected but PSRFITS file metadata must
        be edited appropriately as well and requires the following input:

        Parameters
        ----------

        signal [class] : signal type class (currently only filterbank is supported)
                        used to get the data array to save and other metadata.
        pulsar [class] : pulsar type class used to generate the signal, used for
                        metadata access.
        parfile [string] : path to par file used to generate the polycos. The observing frequency, and observatory will
                            come from the par file. If not provided, a basic par file will be
                            created and saved along with the PSRFITS file.
        MJD_start [float] : Start MJD of the polyco. Should start no later than the beginning of the observation.
        segLength [float] : Length in minutes of the range covered by the polycos generated. Default is 60 minutes.
        ref_MJD [float] : initial time to reference the observations to (MJD). This value
                          should be the start MJD (fraction if necessary) of the first file,
                          default is 56000.0.
        inc_len [float] : time difference (days) between reference MJD and new phase connected
                          MJD, default is 0 (e.g. no time difference).
        usePINT [bool] : Method used to generate polycos. Currently only PINT is supported.
        eq_wts [bool] : If `True` (default), replaces the data weights so that each subintegration and
                        frequency channel have an equal weight in the file. If `False`, just copies the
                        weights from the template file.
        """

        """
        # May come back to this later...
        if self._fits_mode == 'copy':
            pass
        elif self._fits_mode == 'manual':
            pass
        elif self._fits_mode == 'auto':
            pass
        """
        # Check the inclination length
        if inc_len == 0.0:
            inc_len = MJD_start - ref_MJD

        # If mode not search, set nsblk to 1
        if self.obs_mode != 'SEARCH':
            self.nsblk = 1
        
        # Make sure PSRFITS.nsubint and signal.nsub are the same
        if self.nsubint != signal.nsub:
            # Raise a warning
            warnings.warn("PSRFITS.nsubint != SIGNAL.nsub, using SIGNAL.nsub...")
            self._nsubint = signal.nsub
            
        # We need to appropriatly shape the signal for the fits file
        stop = self.nbin*self.nsubint
        sim_sig = signal.data[:,:stop].astype('>i2')
        # out arrays
        Out = np.zeros((self.nsubint, self.npol, self.nchan, self.nbin))
        print(np.shape(Out))
        # We assign the data in the appropriate shape
        for ii in range(self.nsubint):
            idx0 = 0 + ii*self.nbin
            idxF = idx0 + self.nbin
            Out[ii,0,:,:] = sim_sig[:,idx0:idxF]

        self.copy_psrfit_BinTables()
        # We can currently only make total intensity data
        for ii in range(self.nsubint):
            self.file.HDU_drafts['SUBINT'][ii]['DATA'] = Out[ii,0,:,:]
            self.file.HDU_drafts['SUBINT'][ii]['DAT_FREQ'] = signal.dat_freq.value
            # Can't get values from subints that don't exists.
            # If more subints than in initial file, use the last set of values
            if ii >= self.file.fits_template['SUBINT'].get_nrows():
                qq = self.file.fits_template['SUBINT'].get_nrows()-1
            else:
                qq = ii
            if eq_wts:
                # Get the shapes of the wieghts, scales, and offs arrays, assumes we want to reset these to all be equal
                if len(np.shape(self.file.fits_template['SUBINT'][0]['DAT_SCL'])) > 1:
                    scale_shape = np.shape(self.file.fits_template['SUBINT'][qq]['DAT_SCL'][:,:self.nchan*self.npol])
                    offs_shape = np.shape(self.file.fits_template['SUBINT'][qq]['DAT_OFFS'][:,:self.nchan*self.npol])
                    weight_shape = np.shape(self.file.fits_template['SUBINT'][qq]['DAT_WTS'][:,:self.nchan])
                else:
                    scale_shape = np.shape(self.file.fits_template['SUBINT'][qq]['DAT_SCL'])
                    offs_shape = np.shape(self.file.fits_template['SUBINT'][qq]['DAT_OFFS'])
                    weight_shape = np.shape(self.file.fits_template['SUBINT'][qq]['DAT_WTS'])
                # Now assign the values
                #print(scale_shape, offs_shape, weight_shape)
                self.file.HDU_drafts['SUBINT'][ii]['DAT_SCL'] = np.ones(scale_shape)
                self.file.HDU_drafts['SUBINT'][ii]['DAT_OFFS'] = np.zeros(offs_shape)
                self.file.HDU_drafts['SUBINT'][ii]['DAT_WTS'] = np.ones(weight_shape)
            else:
                self.file.HDU_drafts['SUBINT'][ii]['DAT_SCL'] = self.file.fits_template['SUBINT'][qq]['DAT_SCL'][:,:self.nchan*self.npol]
                self.file.HDU_drafts['SUBINT'][ii]['DAT_OFFS'] = self.file.fits_template['SUBINT'][qq]['DAT_OFFS'][:,:self.nchan*self.npol]
                self.file.HDU_drafts['SUBINT'][ii]['DAT_WTS'] = self.file.fits_template['SUBINT'][qq]['DAT_WTS'][:,:self.nchan]

        # Phase connect and adjust required data.
        if parfile == None:
            print("No parfile provided, creating par file %s_sim.par" % (pulsar.name))
            make_par(signal, pulsar, outpar = "%s_sim.par" % (pulsar.name))
            parfile = "%s_sim.par" % (pulsar.name)
        # generate the polyco parameters
        polyco_dict = self._gen_polyco(parfile, MJD_start, segLength = segLength, ncoeff = 15, \
                   maxha=12.0, method="TEMPO", numNodes=20, usePINT = usePint)
        # generate the primary header and subint header parameters
        primary_dict, subint_dict = self._gen_metadata(signal, pulsar, ref_MJD = ref_MJD, inc_len = inc_len)
        # Add values to the subint_dict to fix the metadata, first polarization
        subint_dict['POL_TYPE'] = 'AA+BB'
        # Then the bandwidth
        subint_dict['CHAN_BW'] = self.chan_bw.value
        # Get TSUBINT length
        subint_dict['TSUBINT'] = np.repeat(self.tsubint.value, self.nsubint)
        # Change TBIN
        subint_dict['TBIN'] = pulsar.period.value/self.nbin
        # Change TBIN
        subint_dict['DM'] = signal.dm.value
        # Change TBIN
        subint_dict['NBIN'] = self.nbin
        # Now edit the header parameters with the new phase connected values
        self._edit_psrfits_header(polyco_dict, subint_dict, primary_dict)


        # Now we actually write out the files
        self.file.write_psrfits(hdr_from_draft=True)
        # Close the file so it doesn't take up memory or get confused with another file.
        self.file.close()
        print("Finished writing and saving the file")


    def append(self, signal):
        """Method for appending data to an already existing PSS signal file.
        """
        raise NotImplementedError()

    def load(self):
        """Method for loading saved PSS signal files. These files will have an
        additional BinTable extension, 'PSRSIGSIM', that only contains a header
        with the various PsrSigSim parameters written for references.
        """
        raise NotImplementedError()

    def make_signal_from_psrfits(self):
        """Method to make a signal from the PSRFITS file given as the template.
        For subintegrated data will assume the initial period is the pulsar
        period given in the PSRPARAM header.

        TODO: Currently does not support generating 'SEARCH' mode data from
            a psrfits file

        Parameters
        ----------

        Returns
        -------

        psrsigsim.Signal
        """
        self._fits_mode = 'copy'
        self.get_signal_params()

        if self.obs_mode == 'PSR':
            # Get correct period value from template fits options
            if self.pfit_dict['F'] is None and self.pfit_dict['F0'] is not None:
                s_rate = self.pfit_dict['F0']*self.nbin*10**-6 # in MHz
            elif self.pfit_dict['F'] is not None and self.pfit_dict['F0'] is None:
                s_rate = self.pfit_dict['F']*self.nbin*10**-6 # in MHz
            elif self.pfit_dict['F'] is not None and self.pfit_dict['F0'] is not None:
                s_rate = self.pfit_dict['F0']*self.nbin*10**-6 # in MHz
            else:
                msg = "No pulsar frequency defined in input fits file."
                raise ValueError(msg)
        else:
            s_rate = (1/self.tbin).to('MHz').value

        S = FilterBankSignal(fcent=self.obsfreq.value,
                   bandwidth=self.obsbw.value,
                   Nsubband=self.nchan,
                   sample_rate=s_rate,
                   dtype=np.float32,
                   fold = True,
                   sublen=self.tsubint)

        S._dat_freq = make_quant(self._get_pfit_bin_table_entry('SUBINT', 'DAT_FREQ'), 'MHz')
        S._dm = make_quant(self.pfit_dict['DM'], 'pc/cm^3')

        return S

    def copy_psrfit_BinTables(self, ext_names='all'):
        """Method to copy BinTables from the PSRFITS file given as the template.

        Parameters
        ----------

        ext_names : list, 'all'
            List of BinTable Extensions to copy. Defaults to all, but does not
            copy DATA array in SUBINT BinTable.
        """
        if ext_names == 'all':
            ext_names = self.file.draft_hdr_keys[1:]

        ext_names.remove('SUBINT')
        for ky in ext_names:
            self.file.copy_template_BinTable(ky)

        self.file.set_subint_dims(nbin=self.nbin, nsblk=self.nsblk,
                                  nchan=self.nchan, nsubint=self.nsubint,
                                  npol=self.npol,
                                  data_dtype=self.dtypes['DATA'][0],
                                  obs_mode=self.pfit_dict['OBS_MODE'])
        # Make an array of the desired size, don't copy from template file
        subint_draft = self.file.make_HDU_rec_array(self.file.nsubint, self.file.subint_dtype)
        self.file.HDU_drafts['SUBINT'] = subint_draft


    def to_txt(self):
        """Convert file to txt file.
        """
        raise NotImplementedError()

    def to_psrfits(self):
        """Convert file to PSRFITS file.
        """
        return NotImplementedError()

    def set_sky_info(self):
        """Enter various astronomical and observing data into PSRFITS draft."""
        raise NotImplementedError()

    def _calc_psrfits_dims(self, signal):
        """
        Calculate the needed dimensions of a PSRFITS file from a PSS.signal
        object.
        """
        raise NotImplementedError()

    def get_signal_params(self, signal = None):
        """
        Calculate/retrieve the various parameters to make a PSS signal object
        from a given PSRFITS file.

        if signal is given as a Signal() class object, then the values
        will be taken from the signal class instead of a given PSRFITS
        file.
        """
        # get paramters from fits file
        if signal == None:
            self._make_psrfits_pars_dict()
            self.nchan = self.pfit_dict['NCHAN']
            self.tbin = self.pfit_dict['TBIN']
            self.nbin = self.pfit_dict['NBIN']
            self.npol = self.pfit_dict['NPOL']
            self.nrows = self.pfit_dict['NAXIS2']
            self.nsblk = self.pfit_dict['NSBLK']
            self.obsfreq = self.pfit_dict['OBSFREQ']
            self.obsbw = self.pfit_dict['OBSBW']
            self.chan_bw = self.pfit_dict['CHAN_BW']
            self.stt_imjd = self.pfit_dict['STT_IMJD'] # start MJD of obs
            self.stt_smjd = self.pfit_dict['STT_SMJD'] # start second of obs
            self.tsubint = self.pfit_dict['TSUBINT'] # length of subint in seconds

            if self.obs_mode=='PSR':
                self._nsubint = self.nrows
            else:
                self._nsubint = None
        # get parameters from signal class
        else:
            self._make_psrfits_pars_dict()
            self.nchan = signal.Nchan
            self.tbin = 1.0/signal.samprate
            self.nbin = int(signal.nsamp/signal.nsub)
            self.npol = signal.Npols # self.pfit_dict['NPOL']
            self.nrows = signal.nsub
            self.nsblk = self.pfit_dict['NSBLK']
            self.obsfreq = signal.fcent
            self.obsbw = signal.bw
            self.chan_bw = signal.bw / signal.Nchan
            #self.stt_imjd = self.pfit_dict['STT_IMJD'] # start MJD of obs
            #self.stt_smjd = self.pfit_dict['STT_SMJD'] # start second of obs
            self.tsubint = signal.sublen # length of subint in seconds

            if self.obs_mode=='PSR':
                self._nsubint = self.nrows
            else:
                self._nsubint = None


    def _make_psrfits_pars_dict(self):
        """
        Make parameters from PSRFITS file template into a dictionary for
        initializing PsrSigSim signal object.
        """
        self.pfit_dict = {}

        for extname in self.pfit_pars.keys():
            for ky in self.pfit_pars[extname]:
                if 'DAT' in ky:
                    val = self._get_pfit_bin_table_entry('SUBINT', ky)
                elif 'TSUBINT' in ky:
                    val = self._get_pfit_bin_entry('SUBINT', ky)
                elif extname == "PSRPARAM":
                    val = self._get_pfit_psrparam(extname, ky)
                else:
                    val = self._get_pfit_hdr_entry(extname,ky)
                if isinstance(val,str) or isinstance(val,bytes):
                    val = val.strip()

                self.pfit_dict.update({ky:val})
        idx = self.file.draft_hdr_keys.index('SUBINT')
        dtypes = self.file.get_HDU_dtypes(self.file.fits_template[idx])

        self.dtypes = dict([(item[0],item[1]) if len(item)==2
                            else (item[0],(item[1],item[2]))
                            for item in dtypes])

    def _get_pfit_hdr_entry(self, extname, key):
        """Retrieve a single header entry from PSRFITS file."""
        idx = self.file.draft_hdr_keys.index(extname)
        return self.file.fits_template[idx].read_header()[key]

    def _get_pfit_bin_table_entry(self, extname, key, row=0):
        """Retrieve a single header entry from PSRFITS file."""
        idx = self.file.draft_hdr_keys.index(extname)
        try:
            return self.file.fits_template[idx][key][row]
        except:
            return self.file.fits_template[idx][key][row][0]

    def _get_pfit_bin_entry(self, extname, key, row=0):
        """Retrieve a single header entry from PSRFITS file.
        Different from get_pfit_bin_table_entry, this gets just
        single value parameters (e.g. TSUBINT), not arrays or list values.
        """
        idx = self.file.draft_hdr_keys.index(extname)
        return self.file.fits_template[idx][key][row]

    def _get_pfit_psrparam(self, extname, param):
        """Retrieve a single value from the PSRPARAM header. This
        has a different format than the SUBINT or PRIMARY headers.
        """
        idx = self.file.draft_hdr_keys.index(extname)
        for val in self.file.fits_template[idx][:]:
            if param == val[0].split()[0].decode("utf-8"):
                return np.float64(val[0].split()[1].decode("utf-8").replace("D","E"))

    #### Define various PSRFITS parameters
    @property
    def tbin(self):
        return self._tbin

    @tbin.setter
    def tbin(self, value):
        self._tbin = make_quant(value,'s')

    @property
    def npol(self):
        return self._npol

    @npol.setter
    def npol(self, value):
        self._npol = value

    @property
    def nchan(self):
        return self._nchan

    @nchan.setter
    def nchan(self, value):
        self._nchan = value

    @property
    def nsblk(self):
        return self._nsblk

    @nsblk.setter
    def nsblk(self, value):
        self._nsblk = value

    @property
    def nbin(self):
        return self._nbin

    @nbin.setter
    def nbin(self, value):
        self._nbin = value

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, value):
        self._nrows = value
    
    @property
    def nsubint(self):
        return self._nsubint

    @nsubint.setter
    def nsubint(self, value):
        self._nsubint = value

    @property
    def obsfreq(self):
        return self._obsfreq

    @obsfreq.setter
    def obsfreq(self, value):
        self._obsfreq = make_quant(value,'MHz')

    @property
    def obsbw(self):
        return self._obsbw

    @obsbw.setter
    def obsbw(self, value):
        self._obsbw = make_quant(value,'MHz')

    @property
    def chan_bw(self):
        return self._chan_bw

    @chan_bw.setter
    def chan_bw(self, value):
        self._chan_bw = make_quant(value,'MHz')

    @property
    def stt_imjd(self):
        return self._stt_imjd

    @stt_imjd.setter
    def stt_imjd(self, value):
        self._stt_imjd = make_quant(value, 'day')

    @property
    def stt_smjd(self):
        return self._stt_smjd

    @stt_smjd.setter
    def stt_smjd(self, value):
        self._stt_smjd = make_quant(value, 'second')

    @property
    def tsubint(self):
        return self._tsubint

    @tsubint.setter
    def tsubint(self, value):
        self._tsubint = make_quant(value, 'second')
