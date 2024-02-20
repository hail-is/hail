.. _sec-saige_api:

==========
Python API
==========

This is the API documentation for the Hail Batch implementation of SAIGE.

Use ``import hailtop.saige`` to access this functionality.


.. currentmodule:: hailtop.saige


saige
~~~~~

.. autosummary::
    :toctree: saige-api/saige/
    :nosignatures:
    :template: class.rst

    saige.SaigeConfig


.. autosummary::
    :toctree: saige-api/saige/
    :nosignatures:

    saige.extract_phenotypes


config
~~~~~~

.. autosummary::
    :toctree: saige-api/config/
    :nosignatures:
    :template: class.rst

    config.CheckpointConfigMixin
    config.JobConfigMixin


phenotype
~~~~~~~~~

.. autosummary::
    :toctree: saige-api/phenotype/
    :nosignatures:
    :template: class.rst

    phenotype.Phenotype
    phenotype.PhenotypeConfig
    phenotype.SaigePhenotype


variant_chunk
~~~~~~~~~~~~~

.. autosummary::
    :toctree: saige-api/variant_chunk/
    :nosignatures:
    :template: class.rst

    variant_chunk.VariantChunk


steps
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


constants
~~~~~~~~~

.. autosummary::
    :toctree: saige-api/constants/
    :nosignatures:
    :template: class.rst

    constants.SaigeAnalysisType
