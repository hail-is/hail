========================
Install Hail on Mac OS X
========================

- Install Java 8 or Java 11. We recommend using a
  `packaged installation from Azul <https://www.azul.com/downloads/?version=java-8-lts&os=macos&package=jdk&show-old-builds=true>`__
  (make sure the OS version and architecture match your system) or using `Homebrew <https://brew.sh/>`__:

  .. code-block::

    brew tap homebrew/cask-versions
    brew install --cask temurin8

- Install Python 3.8 or later. We recommend `Miniconda <https://docs.conda.io/en/latest/miniconda.html#macosx-installers>`__.
- Open Terminal.app and execute ``pip install hail``. If this command fails with a message about "Rust", please try this instead: ``pip install hail --only-binary=:all:``.
- `Run your first Hail query! <try.rst>`__
