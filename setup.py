#! /usr/bin/env python
"""An MNE compatible package for processing near-infrared spectroscopy data."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('mne_nirs', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'mne-nirs'
DESCRIPTION = 'An MNE compatible package for processing near-infrared spectroscopy data.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Robert Luke'
MAINTAINER_EMAIL = 'robert.luke@mq.edu.au'
URL = 'https://mne.tools/mne-nirs/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mne-tools/mne-nirs'
VERSION = __version__
INSTALL_REQUIRES = ['numpy>=1.11.3', 'scipy>=0.17.1', 'mne>=0.21.0'],
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3',
               ]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      keywords='neuroscience neuroimaging fNIRS NIRS brain',
      packages=find_packages(),
      project_urls={
          'Documentation': 'https://mne.tools/mne-nirs/',
          'Source': 'https://github.com/mne-tools/mne-nirs/',
          'Tracker': 'https://github.com/mne-tools/mne-nirs/issues/',
      },
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
