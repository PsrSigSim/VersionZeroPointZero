#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `psrsigsim` package."""

import pytest
import unittest

import psrsigsim as PSS


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
    pass


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    pass

class test_PSS_classes(unittest.TestCase):

    def test_signal(self):
        """Test Signal Instantiation"""
        S1 = PSS.Signal()

    def test_pulsar(self):
        """Test Pulsar Class"""
        S1 = PSS.Signal()
        P1 = PSS.Pulsar(S1)
        P1.make_pulses()
