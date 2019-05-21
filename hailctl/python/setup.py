from setuptools import setup

setup(name='hailctl',
      version='devel',
      description='Manage and monitor Hail deployments.',
      url='https://github.com/hail-is/hail',
      author='Hail Team',
      author_email='hail@broadinstitute.org',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='hail google cloud dataproc spark jupyter',
      packages=['hailctl'],
      entry_points={
          'console_scripts': [
              'hailctl = hailctl.__main__:main'
          ]
      },
      )
