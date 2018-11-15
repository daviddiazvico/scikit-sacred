"""
@author: David Diaz Vico
@license: MIT
"""

from setuptools import find_packages, setup


setup(name='scikit-sacred',
      packages=find_packages(),
      version='0.1.2',
      description='Scikit-learn-compatible Sacred experiments',
      long_description=open('README.md', 'r').read(),
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-sacred',
      download_url='https://github.com/daviddiazvico/scikit-sacred/archive/v0.1.2.tar.gz',
      keywords=['sacred', 'scikit-learn'],
      classifiers=['Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],
      install_requires=['sacred'],
      setup_requires=['pytest-runner'],
      tests_require=['scikit-learn', 'coverage', 'pytest', 'pytest-cov'])
