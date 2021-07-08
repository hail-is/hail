========================
Install Hail on Mac OS X
========================

- Install Java 8. We recommend using a
  `packaged installation from Azul <https://www.azul.com/downloads/?version=java-8-lts&os=macos&package=jdk&show-old-builds=true>`__
  (make sure the OS version and architecture match your system) or using `Homebrew <https://brew.sh/>`__:

  .. code-block::

    brew cask install adoptopenjdk8
    brew install --cask adoptopenjdk8

- Install Python 3.6+. We recommend `Miniconda <https://docs.conda.io/en/latest/miniconda.html#macosx-installers>`__.
- Open Terminal.app and execute ``pip install hail``.
- `Run your first Hail query! <try.rst>`__
