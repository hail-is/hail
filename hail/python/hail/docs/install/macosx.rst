========================
Install Hail on Mac OS X
========================

- Install Java 11. We recommend using a `packaged installation from Azul
  <https://www.azul.com/downloads/?version=java-11-lts&os=macos&package=jdk&show-old-builds=true>`__
  (make sure the OS version and architecture match your system) or using `Homebrew
  <https://brew.sh/>`__:

  .. code-block::

    brew tap homebrew/cask-versions
    brew install --cask temurin8

  You *must* pick a Java installation with a compatible architecture. If you have an Apple M1 or M2
  you must use an "arm64" Java, otherwise you must use an "x86_64" Java. You can check if you have
  an M1 or M2 either in the "Apple Menu > About This Mac" or by running ``uname -m`` Terminal.app.

- Install Python 3.9 or later. We recommend `Miniconda <https://docs.conda.io/en/latest/miniconda.html#macosx-installers>`__.
- Open Terminal.app and execute ``pip install hail``. If this command fails with a message about "Rust", please try this instead: ``pip install hail --only-binary=:all:``.
- `Run your first Hail query! <try.rst>`__

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
hailctl Autocompletion (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Install autocompletion with ``hailctl --install-completion zsh``
- Ensure this line is in your zsh config file (~/.zshrc) and then reload your terminal.

  .. code-block::

    autoload -Uz compinit && compinit
