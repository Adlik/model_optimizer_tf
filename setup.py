#!/usr/bin/env python3

# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model optimizer tf.
"""

from setuptools import find_packages, setup
from pkg_resources import DistributionNotFound, get_distribution
_VERSION = '0.0.0'

_REQUIRED_PACKAGES = [
    'requests==2.25.0',
    'tensorflow==2.3.0',
    'jsonschema==3.1.1',
    'networkx==2.4',
    'mpi4py==3.1.3',
    'horovod==0.24.0',
    'tf2cv==0.0.16',
    'PyYAML==5.3.1',
    'types-PyYAML',
    'types-pkg_resources',
    'types-requests'
]

_TEST_REQUIRES = [
    'bandit',
    'pytest-cov',
    'pytest-flake8',
    'pytest-mypy',
    'pytest-pylint',
    'pytest-xdist'
]


def get_dist(pkgname):
    """
    Get distribution
    :param pkgname: str, package name
    :return:
    """
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


if get_dist('tensorflow') is None and get_dist('tensorflow-gpu') is not None:
    _REQUIRED_PACKAGES.remove('tensorflow==2.3.0')

setup(
    name="model_optimizer_tf",
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
        'model_optimizer_tf': ['**/*.json',
                            'pruner/scheduler/uniform_auto/*.yaml',
                            'pruner/scheduler/uniform_specified_layer/*.yaml',
                            'pruner/scheduler/distill/*.yaml']
    },

)
