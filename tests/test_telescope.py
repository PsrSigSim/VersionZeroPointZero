#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `psrsigsim.telescope` module."""

import pytest
import psrsigsim as pss
import psrsigsim.signal as sig
import psrsigsim.pulsar as psr
import psrsigsim.telescope as telescope

from psrsigsim.signal.fb_signal import FilterBankSignal
from psrsigsim.signal.bb_signal import BasebandSignal
from psrsigsim.pulsar.pulsar import Pulsar
from psrsigsim.telescope.telescope import Telescope
from psrsigsim.telescope.receiver import Receiver, _flat_response, response_from_data
from psrsigsim.telescope.backend import Backend
from psrsigsim.utils.utils import make_quant


@pytest.fixture
def signal():
    """
    Fixture signal class
    """
    fbsig = FilterBankSignal(1400,400, fold=False, sample_rate = (1.0/0.005)*2048*10**-6)
    return fbsig

@pytest.fixture
def bb_signal():
    """
    Fixture signal class
    - Makes a baseband signal
    """
    bbsig = BasebandSignal(1400,400)
    return bbsig

@pytest.fixture
def subint_signal():
    """
    Fixture for signal class
    - Makes a subintegrated signal
    """
    fbsubsig = FilterBankSignal(1400,400,fold=True)
    return fbsubsig

@pytest.fixture
def pulsar():
    """
    Fixture pulsar class
    """
    period = make_quant(5,'ms')
    return Pulsar(period,10,name='J1746-0118')


@pytest.fixture
def tscope():
    """
    Fixture telescope class
    """
    scope = Telescope(20.0, area=30.00, name="Twenty_Meter")
    return scope

@pytest.fixture
def receiver():
    """
    Fixture receiver class
    """
    rcvr = Receiver(fcent=1400,bandwidth=400,name="Lband")
    return rcvr

@pytest.fixture
def backend():
    """
    Fixture backend class
    """
    smprte = make_quant(81.92, "microsecond")
    bckend = Backend(samprate=(1.0/smprte).to("MHz"), name="Cyborg")
    return bckend

def test_backend(signal, pulsar, backend):
    """
    Test for backend class.
    """
    backend_names = backend.__repr__()
    assert(backend.name == "Cyborg")
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(signal,tobs)
    fold_sig = backend.fold(signal, pulsar)
    
def test_rcvr_init():
    """
    Test for initializing the receiver class.
    """
    with pytest.raises(ValueError):
        rcvr = Receiver(fcent=None,bandwidth=400,name="Lband")
        rcvr2 = Receiver(response = _flat_response(1400, 400), fcent=None,bandwidth=400,name="Lband")
    with pytest.raises(NotImplementedError):
        rcvr3 = Receiver(response = _flat_response(1400, 400), name="Lband")

def test_rcvr(signal, pulsar, receiver):
    """
    Tests for the receiver class.
    """
    rcvr_names = receiver.__repr__()
    assert(receiver.name == "Lband")
    assert(callable(receiver.response))
    assert(receiver.fcent.value == 1400.0)
    assert(receiver.bandwidth.value == 400.0)
    # Test Tenv checks
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(signal,tobs)
    receiver.radiometer_noise(signal, pulsar, gain=1, Tsys=None, Tenv=None)
    with pytest.raises(ValueError):
        receiver.radiometer_noise(signal, pulsar, gain=1, Tsys=1.0, Tenv=1.0)
    receiver.radiometer_noise(signal, pulsar, gain=1, Tsys=None, Tenv=make_quant(2.0,'K'))
    # Test get noise failure
    with pytest.raises(NotImplementedError):
        signal._sigtype = None
        receiver.radiometer_noise(signal, pulsar, gain=1, Tsys=None, Tenv=None)
    # Test baseband radiometer noise
    # TODO: Put test here

def test_rcvr_ampnoise(signal, pulsar, receiver):
    """
    Testing making amplitude noise for baseband signal.
    """
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(signal,tobs)
    Tsys = make_quant(35.0,'K')
    gain = 1.0
    noise = receiver._make_amp_noise(signal, Tsys, gain, pulsar)

def test_response():
    """
    Test the different response functions.
    """
    response1 = _flat_response(1400, 400)
    with pytest.raises(NotImplementedError):
        response2 = response_from_data([300,350,400], [1,1,2,3,2,1,1])
    
def test_obs(tscope, receiver, backend, signal, pulsar):
    """
    Test adding a system to the telescope and observing
    """
    tscope.add_system(name="Twnty_M", receiver=receiver, backend=backend)
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(signal,tobs)
    tscope.observe(signal, pulsar, system="Twnty_M", noise=False)

def test_noise(tscope, receiver, backend, signal, pulsar):
    """
    Test adding a system to the telescope and observing with noise
    """
    tscope.add_system(name="Twnty_M", receiver=receiver, backend=backend)
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(signal,tobs)
    tscope.observe(signal, pulsar, system="Twnty_M", noise=True)
    
def test_subint_obs(tscope, receiver, backend, subint_signal, pulsar):
    """
    Test adding a system to the telescope and observing with subint signal
    """
    tscope.add_system(name="Twnty_M", receiver=receiver, backend=backend)
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(subint_signal,tobs)
    tscope.observe(subint_signal, pulsar, system="Twnty_M", noise=False)

def test_subint_noise(tscope, receiver, backend, subint_signal, pulsar):
    """
    Test adding a system to the telescope and observing with noise and subint
    signal
    """
    tscope.add_system(name="Twnty_M", receiver=receiver, backend=backend)
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(subint_signal,tobs)
    tscope.observe(subint_signal, pulsar, system="Twnty_M", noise=True)

def test_bb_obs(tscope, receiver, backend, bb_signal, pulsar):
    """
    Test adding a system to the telescope and observing with subint signal
    """
    tscope.add_system(name="Twnty_M", receiver=receiver, backend=backend)
    tobs = make_quant(0.01,'s')
    pulsar.make_pulses(bb_signal,tobs)
    with pytest.raises(NotImplementedError):
        tscope.observe(bb_signal, pulsar, system="Twnty_M", noise=False)

def test_sampling(receiver, signal, pulsar):
    """
    Test sampling rate comparisons.
    Lines to Test: 89, 111, 136-139
    Not 100% sure some of these lines are being tested appropriately.
    """
    scope1 = Telescope(20.0, area=None, Tsys = 25.0, name="Twenty_Meter")
    # Check dt_tel = dt_sig
    backend1 = Backend(samprate=1.0/make_quant(4.8828125e-06, 's'), name="Cyborg")
    scope1.add_system(name="Twnty_M", receiver=receiver, backend=backend1)
    tobs = make_quant(0.02,'s')
    pulsar.make_pulses(signal,tobs)
    scope1.observe(signal, pulsar, system="Twnty_M", noise=True)
    # Check dt_tel % dt_sig = 0
    backend2 = Backend(samprate=1.0/make_quant(9.765625e-06, 's'), name="Cyborg")
    scope1.add_system(name="Twnty_M", receiver=receiver, backend=backend2)
    scope1.observe(signal, pulsar, system="Twnty_M", noise=True)
    
def test_properties(signal, receiver, backend):
    """
    Test some properties and functions of the telescope class.
    """
    # Test init telescope without area
    scope1 = Telescope(20.0, area=None, Tsys = 25.0, name="Twenty_Meter")
    scope1.add_system(name="Twnty_M", receiver=receiver, backend=backend)
    # Check properties
    scope_names = scope1.__repr__()
    assert(scope1.name == "Twenty_Meter")
    assert(scope1.aperture.value == 20.0)
    assert(scope1.Tsys.value == 25.0)
    with pytest.raises(NotImplementedError):
        scope1.apply_response(signal)
        scope1.rfi()
        scope1.init_signal("Twnty_M")