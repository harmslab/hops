import sys 
if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

from setuptools import setup, find_packages, Extension
import numpy
from Cython.Distutils import build_ext

setup(name="peplearn",
      packages=find_packages(),
      version='0.0.2',
      description="machine learning for peptide binding using physiochemical properties",
      long_description=open("README.rst").read(),
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmslab/peplearn',
      download_url='https://github.com/harmslab/peplearn/tarball/0.0.2',
      install_requires=["numpy","localcider"],
      package_data={"peplearn":["*.json","features/data/*.json"]},
      zip_safe=False,
      classifiers=['Programming Language :: Python'])
