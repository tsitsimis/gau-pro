from setuptools import setup
# from distutils.core import setup

setup(name='carly',
      packages=['carly'],
      version='0.1.1',
      description='A Python implementation of the Gaussian Processes framework',
      author='Theodore Tsitsimis',
      author_email='th.tsitsimis@gmail.com',
      url='https://github.com/tsitsimis/carly',
      download_url='https://github.com/tsitsimis/carly/archive/0.1.1.tar.gz',
      keywords=['gaussian-process', 'machine-learning'],
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3.4'
      ],
      install_requires=[
          'numpy',
      ],
      zip_safe=False
      )
