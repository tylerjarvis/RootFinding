#try:
#    from setuptools import setup
#except ImportError:
#    from distutils.core import setup
from distutils.core import setup

#from pip.req import parse_requirements

config = {
        'description':"Stable numerical commputation of Groebner bases.",
        'packages':['groebner'],
        'scripts':['bin/groebner']
        }

setup(**config)
