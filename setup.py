#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy


with open('README.md') as readme_file:
    readme = readme_file.read()

ext_modules = [Extension('cgmath',
                         ['cellgrid/cgmath.pyx'],
                         include_dirs=[numpy.get_include()])]

requirements = [
    'numpy>=1.4.0',
]

test_requirements = [
    'numpy>=1.4.0',
    'nose>=0.10',
]

setup(
    name='cellgrid',
    version='0.1.0',
    description="CellGrids for coordinate analysis",
    long_description=readme + '\n\n',
    author="Richard J Gowers",
    author_email='richardjgowers@gmail.com',
    url='https://github.com/richardjgowers/cellgrid',
    packages=[
        'cellgrid',
    ],
    package_dir={'cellgrid':
                 'cellgrid'},
    cmdclass = {'build_ext': build_ext},
    ext_package = 'cellgrid',
    ext_modules = ext_modules,
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='cellgrid',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
