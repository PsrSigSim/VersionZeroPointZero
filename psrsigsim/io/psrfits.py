# -*- coding: utf-8 -*-
# encoding=utf8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import pdat
from .file import BaseFile
from ..utils import make_quant
from ..signal import Signal
from ..signal import FilterBankSignal

__all__ = ["PSRFITS"]

class PSRFITS(BaseFile):
    """A class for saving PsrSigSim signals as PSRFITS standard files.
    path: name and path of new psrfits file that will be saved
    obs_mode: what type of observation is the data, SEARCH, PSR, etc. 
    template: the path and name of the template fits file that will be loaded
    copy_template: Does nothing?
    fits_mode: how we want to save the data, right now just 'copy' is valid
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
                                    "TSUBINT",] # TTYPE2 is the subint length
                           }

        #Might not need this, since we can calculate from other terms.
        if self.obs_mode=='SEARCH':
            self.pfit_pars['SUBINT'].append('TDIM17')
        elif self.obs_mode=='PSR':
            self.pfit_pars['SUBINT'].append('TDIM20')

    def save(self, signal):
        """Save PSS signal file to disk.
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
        # We can only save single polarization
        self.file.set_draft_header('SUBNT',{'POL_TYPE':'AA+BB'})
        

        raise NotImplementedError()

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

        Parameters
        ----------

        Returns
        -------

        psrsigsim.Signal
        """
        self._fits_mode = 'copy'
        self._get_signal_params()
        
        if self.obs_mode == 'PSR':
            ObsTime = self.tsubint*self.nrows
        else:
            ObsTime = self.tbin*self.nbin*self.nsblk*self.nrows

        #TODO Delete calls to .value when integrated with new API.
        #TODO Change call to FilterBank for new API.
        """S = Signal(f0=self.obsfreq.value,
                   bw=self.obsbw.value,
                   Nf=self.nchan,
                   f_samp=(1/self.tbin).to('MHz').value,
                   ObsTime=ObsTime.to('ms').value,
                   data_type='float32',
                   SignalType='intensity',
                   mode='simulate',
                   clean_mode=True)
        """
        S = FilterBankSignal(fcent=self.obsfreq.value,
                   bandwidth=self.obsbw.value,
                   Nsubband=self.nchan,
                   sample_rate=(1/self.tbin).to('MHz').value,
                   #ObsTime=ObsTime.to('ms').value,
                   dtype=np.float32,
                   subint=True,
                   sublen=self.tsubint)

        S.freq_Array = self._get_pfit_bin_table_entry('SUBINT', 'DAT_FREQ')

        return S

    def copy_psrfit_BinTables(self, ext_names='all', copy_SUBINT_nonDATA=True):
        """Method to copy BinTables from the PSRFITS file given as the template.

        Parameters
        ----------

        ext_names : list, 'all'
            List of BinTable Extensions to copy. Defaults to all, but does not
            copy DATA array in SUBINT BinTable.

        copy_SUBINT_nonDATA : bool, optional
            If True all of the contents of the `SUBINT` BinTable will be copied
            except for the DATA array. This includes the single floats as well
            as TSUBINT, OFFS_SUB, LST_SUB, RA_SUB, DEC_SUB, GLON_SUB, GLAT_SUB,
            FD_ANG, POS_ANG, PAR_ANG, TEL_AZ, TEL_ZEN and others. Is set to
            False if ext_names does not include 'SUBINT' or not equal to 'all'.
            Does not set DAT_FREQ, DAT_SCL, DAT_WTS, DAT_OFFS.
        """
        if ext_names == 'all':
            ext_names = self.file.draft_hdr_keys[1:]
        if 'SUBINT' not in ext_names:
            copy_SUBINT_nonDATA = False

        ext_names.remove('SUBINT')
        for ky in ext_names:
            self.file.copy_template_BinTable(ky)

        if copy_SUBINT_nonDATA:
            self.file.set_subint_dims(nbin=self.nbin, nsblk=self.nsblk,
                                      nchan=self.nchan, nsubint=self.nrows,
                                      npol=self.npol,
                                      data_dtype=self.dtypes['DATA'][0],
                                      obs_mode=self.pfit_dict['OBS_MODE'])

            self.file.copy_template_BinTable(ext_name='SUBINT',
                                            cols=self.file.single_subint_floats)


    def to_txt(self):
        """Convert file to txt file.
        """
        raise NotImplementedError()

    def to_psrfits(self):
        """Convert file to PSRFITS file.
        """
        return self

    def set_sky_info(self):
        """Enter various astronomical and observing data into PSRFITS draft."""
        raise NotImplementedError()

    def _calc_psrfits_dims(self, signal):
        """
        Calculate the needed dimensions of a PSRFITS file from a PSS.signal
        object.
        """
        self.nchan = signal.Nf
        Nt = signal.Nt

    def _get_signal_params(self):
        """
        Calculate/retrieve the various parameters to make a PSS signal object
        from a given PSRFITS file.

        Returns
        -------
        NChan, Nt
        """
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
            self.nsubint = self.nrows
        else:
            self.nsubint = None


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
        return self.file.fits_template[idx][key][row][0]
    
    def _get_pfit_bin_entry(self, extname, key, row=0):
        """Retrieve a single header entry from PSRFITS file.
        Different from get_pfit_bin_table_entry, this gets just
        single value parameters (e.g. TSUBINT), not arrays or list values.
        """
        idx = self.file.draft_hdr_keys.index(extname)
        return self.file.fits_template[idx][key][row]

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
