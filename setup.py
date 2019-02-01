#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Tyler Jarvis",
    author_email='jarvis@math.byu.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="A package for numerical root finding.",
    #install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n',# + history,
    include_package_data=True,
    keywords='RootFinding',
    name='RootFinding',
<<<<<<< HEAD
    packages=find_packages(include=['yroots']),
=======
    packages=find_packages(include=['yroots', 'CHEBYSHEV/TVB_Method']),
>>>>>>> public_code
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tylerjarvis/RootFinding',
    version='0.1.0',
    zip_safe=False,
)
