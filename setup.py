__description__ = \
"""
"""
__version__ = "0.0.2"
import sys 

if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

from setuptools import setup, find_packages, Extension
import numpy
from Cython.Distutils import build_ext

setup(name="peplearn",
      packages=find_packages(),
      version=__version__,
      description="machine learning for peptide binding using physiochemical properties",
      long_description=__description__,
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmslab/peplearn',
      download_url="https://github.com/harmslab/peplearn/archive/{}.tar.gz".format(__version__),
      install_requires=["numpy","localcider"],
      package_data={"peplearn":["*.json","features/data/*.json"]},
      zip_safe=False,
      classifiers=['Programming Language :: Python'])
