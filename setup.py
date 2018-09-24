from setuptools import find_packages, setup


setup(name='scikit-sacred',
      packages=find_packages(),
      version='0.1.1',
      description='Scikit-learn-compatible Sacred experiments',
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-sacred',
      download_url='https://github.com/daviddiazvico/scikit-sacred/archive/v0.1.1.tar.gz',
      keywords=['sacred', 'scikit-learn'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      install_requires=['sacred', 'scikit-learn'])
