#!/usr/bin/env python

from setuptools import find_packages, setup


description = 'A new approach for representing biological sequences'
REQUIRED_PACKAGES = [
    'gensim==3.4.0',
    'tqdm>=2.0.0',
    'pyfaidx==0.7.1',
    'numpy>=1.18.1'
]

setup(name='biovec',
      version='0.2.7',
      license='MIT',
      description=description,
      long_description=description,
      author="kyu999",
      author_email="kyukokkyou999@gmail.com",
      maintainer="kyu999",
      maintainer_email="kyukokkyou999@gmail.com",
      url='https://github.com/kyu999/biovec',
      packages=find_packages(),
      install_requires=REQUIRED_PACKAGES,
      )
