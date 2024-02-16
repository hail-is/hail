.. _sec-saige_api:

================
SAIGE Python API
================

This is the API documentation for the Hail Batch implementation of SAIGE.

Use ``import hailtop.saige`` to access this functionality.


.. currentmodule:: hailtop.saige


SAIGE
~~~~~

.. autosummary::
    :toctree: saige-api/saige/
    :nosignatures:
    :template: class.rst

    saige.SaigeConfig

Config
~~~~~~

.. autosummary::
    :toctree: saige-api/config/
    :nosignatures:
    :template: class.rst

    config.CheckpointConfigMixin
    config.JobConfigMixin


Steps
~~~~~

.. autosummary::
    :toctree: saige-api/steps/
    :nosignatures:
    :template: class.rst

    steps.CompileAllResultsStep
    steps.CompilePhenotypeResultsStep
    steps.SparseGRMStep
    steps.Step1NullGlmmStep
    steps.Step2SPAStep


Phenotypes
~~~~~~~~~~


Variant Chunks
~~~~~~~~~~~~~~
