#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `psrsigsim.signal` module."""

import pytest
import psrsigsim as pss
from astropy import units as u

# import these fixtures for other module tests

@pytest.fixture
def fb_sig():
    """
    Fixture FilterBankSignal class
    """
    fcent = 1.5 * u.GHz
    bw = 800 * u.MHz
    fbsig = pss.signal.FilterBankSignal(fcent, bw)
    return fbsig

@pytest.fixture
def bb_sig():
    """
    Fixture BasebandSignal class
    """
    fcent = 820 * u.MHz
    bw = 200 * u.MHz
    fbsig = pss.signal.BasebandSignal(fcent, bw)
    return bbsig

@pytest.fixture
def rf_sig():
    """
    Fixture RFSignal class
    """
    fcent = 430 * u.MHz
    bw = 100 * u.MHz
    fbsig = pss.signal.RFSignal(fcent, bw)
    return bbsig

def test_FilterBank():
    # instantiate a FilterBank many ways
    fcent = 1.5 * u.GHz
    bw = 800 * u.MHz
    mySig = pss.signal.FilterBankSignal(fcent, bw)
    
    mySig = pss.signal.FilterBankSignal(fcent, bw, Nsubband=64)
    mySig = pss.signal.FilterBankSignal(fcent, bw, sample_rate=100)
    mySig = pss.signal.FilterBankSignal(fcent, bw, subint=True)

def test_BasebandSignal():
    # instantiate a BasebandSignal many ways
    fcent = 820 * u.MHz
    bw = 200 * u.MHz
    mySig = pss.signal.BasebandSignal(fcent, bw)
    mySig = pss.signal.BasebandSignal(fcent, bw, sample_rate=100)
    mySig = pss.signal.BasebandSignal(fcent, bw, sample_rate=800)

def test_RFSignal():
    # instantiate a RFSignal many ways
    fcent = 430 * u.MHz
    bw = 100 * u.MHz
    mySig = pss.signal.RFSignal(fcent, bw)
    
    mySig = pss.signal.RFSignal(fcent, bw, sample_rate=100)
    mySig = pss.signal.RFSignal(fcent, bw, sample_rate=800)

