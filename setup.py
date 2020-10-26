#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'scipy',
    'matplotlib>=2.0.0',
    'h5py',
    'pdat',
    'pint-pulsar',
]

setup_requirements = [
    'pytest-runner',
    # TODO: put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='psrsigsim',
    version='1.0.0',
    description="The NANOGrav Pulsar Signal Simulator.",
    long_description=readme + '\n\n' + history,
    author="Jeffrey S. Hazboun",
    author_email='jeffrey.hazboun@nanograv.org',
    url='https://github.com/PsrSigSim/PsrSigSim',
    packages=find_packages(include=['psrsigsim',
                                    'psrsigsim.signal',
                                    'psrsigsim.pulsar',
                                    'psrsigsim.telescope',
                                    'psrsigsim.ism',
                                    'psrsigsim.utils',
                                    'psrsigsim.io',
                                    'psrsigsim.simulate']),
    include_package_data=True,
    package_data={'psrsigsim': ['PTA_pulsar_nb_data.txt', 'data/*.par','data/*.npy']},
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='psrsigsim',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
