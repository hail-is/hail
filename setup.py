from setuptools import setup

setup(name='cloudtools',
	  version='1.1.14',
	  description='Collection of utilities for working on the Google Cloud Platform.',
	  url='https://github.com/Nealelab/cloudtools',
	  author='Liam Abbott',
	  author_email='labbott@broadinstitute.org',
	  license='MIT',
	  classifiers=[
	      'Development Status :: 3 - Alpha',
	      'License :: OSI Approved :: MIT License',
	      'Programming Language :: Python :: 2.7'
	  ],
	  keywords='google cloud dataproc spark jupyter hail notebook ipython',
	  packages=['cloudtools'],
	  entry_points={
	      'console_scripts': [
	          'cluster = cloudtools.__main__:main'
	      ]
	  },
	  )
