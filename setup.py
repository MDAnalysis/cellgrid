#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

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
