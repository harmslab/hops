__description__ = \
"""
"""
__version__ = "0.0.2"
import sys 

if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

from setuptools import setup, find_packages
import numpy

setup(name="hops",
      packages=find_packages(),
      version=__version__,
      description="machine learning for peptide binding using physiochemical properties",
      long_description=__description__,
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmslab/hops',
      download_url="https://github.com/harmslab/hops/archive/{}.tar.gz".format(__version__),
      install_requires=["numpy","localcider"],
      package_data={"hops":["*.json","features/data/*.json",
                                "*.json","features/data/util/*.json",
                                "*.txt","features/data/*.txt",
                                "*.txt","features/data/util/*.txt",
                                "*.ipynb","features/data/util/*.ipynb"]},
      zip_safe=False,
      classifiers=['Programming Language :: Python'],
      entry_points = {
            'console_scripts': [
                  'hops_kmerize = hops.console.kmerize:main',
                  'hops_features = hops.console.features:main',
                  'hops_train = hops.console.train:main',
                  'hops_predict = hops.console.predict:main',
                  'hops_stats = hops.console.stats:main',
                  'hops_pred_to_fasta = hops.console.pred_to_fasta:main',
            ]
      })
