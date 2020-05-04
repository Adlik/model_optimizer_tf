#!/usr/bin/env python3

# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model optimizer.
"""

from setuptools import find_packages, setup

_VERSION = '0.0.0'

_REQUIRED_PACKAGES = [
    'requests',
    'tensorflow==2.1.0',
    'jsonschema==3.1.1',
    'horovod==0.19.1'
]

_TEST_REQUIRES = [
    'bandit',
    'pytest-cov',
    'pytest-flake8',
    'pytest-mypy',
    'pytest-pylint',
    'pytest-xdist'
]

setup(
    name="model_optimizer",
    version=_VERSION.replace('-', ''),
    author='ZTE',
    author_email='ai@zte.com.cn',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description=__doc__,
    license='Apache 2.0',
    keywords='optimizer model',
    install_requires=_REQUIRED_PACKAGES,
    extras_require={'test': _TEST_REQUIRES},
    package_data={
        'model_optimizer': ['*.json']
    },

)

