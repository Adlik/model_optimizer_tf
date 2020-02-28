"""
The example packaging script.
"""

import setuptools

setuptools.setup(
    name='example',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    extras_require={
        'test': [
            'bandit',
            'pytest-cov',
            'pytest-flake8',
            'pytest-mypy',
            'pytest-pylint',
            'pytest-xdist'
        ]
    }
)
