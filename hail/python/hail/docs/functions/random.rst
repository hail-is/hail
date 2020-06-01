Random functions
----------------

.. currentmodule:: hail.expr.functions


Hail has several functions that generate random values when invoked. The values
are seeded when the function is called, so calling a random Hail function and
then using it several times in the same expression will yield the same result
each time.

Evaluating the same expression will yield the same value every time, but multiple
calls of the same function will have different results. For example, let `x` be
a random number generated with the function :func:`.rand_unif`:

    >>> x = hl.rand_unif(0, 1)

The value of `x` will not change, although other calls to :func:`.rand_unif`
will generate different values:

    >>> hl.eval(x)  # doctest: +SKIP_OUTPUT_CHECK
    0.5562065047992025

    >>> hl.eval(x)  # doctest: +SKIP_OUTPUT_CHECK
    0.5562065047992025

    >>> hl.eval(hl.rand_unif(0, 1))  # doctest: +SKIP_OUTPUT_CHECK
    0.4678132874101748

    >>> hl.eval(hl.rand_unif(0, 1))  # doctest: +SKIP_OUTPUT_CHECK
    0.9097632224065403

    >>> hl.eval(hl.array([x, x, x]))  # doctest: +SKIP_OUTPUT_CHECK
    [0.5562065047992025, 0.5562065047992025, 0.5562065047992025]

If the three values in the last expression should be distinct, three separate
calls to :func:`.rand_unif` should be made:

    >>> a = hl.rand_unif(0, 1)
    >>> b = hl.rand_unif(0, 1)
    >>> c = hl.rand_unif(0, 1)
    >>> hl.eval(hl.array([a, b, c]))  # doctest: +SKIP_OUTPUT_CHECK
    [0.8846327207915881, 0.14415148553468504, 0.8202677741734825]

Within the rows of a :class:`.Table`, the same expression will yield a
consistent value within each row, but different (random) values across rows:

    >>> table = hl.utils.range_table(5, 1)
    >>> table = table.annotate(x1=x, x2=x, rand=hl.rand_unif(0, 1))
    >>> table.show()  # doctest: +SKIP_OUTPUT_CHECK
    +-------+-------------+-------------+-------------+
    |   idx |          x1 |          x2 |        rand |
    +-------+-------------+-------------+-------------+
    | int32 |     float64 |     float64 |     float64 |
    +-------+-------------+-------------+-------------+
    |     0 | 8.50369e-01 | 8.50369e-01 | 9.64129e-02 |
    |     1 | 5.15437e-01 | 5.15437e-01 | 8.60843e-02 |
    |     2 | 5.42493e-01 | 5.42493e-01 | 1.69816e-01 |
    |     3 | 5.51289e-01 | 5.51289e-01 | 6.48706e-01 |
    |     4 | 6.40977e-01 | 6.40977e-01 | 8.22508e-01 |
    +-------+-------------+-------------+-------------+

The same is true of the rows, columns, and entries of a :class:`.MatrixTable`.

Setting a seed
==============

All random functions can take a specified seed as an argument. This guarantees
that multiple invocations of the same function within the same context will
return the same result, e.g.

    >>> hl.eval(hl.rand_unif(0, 1, seed=0))  # doctest: +SKIP_OUTPUT_CHECK
    0.5488135008937808

    >>> hl.eval(hl.rand_unif(0, 1, seed=0))  # doctest: +SKIP_OUTPUT_CHECK
    0.5488135008937808

This does not guarantee the same behavior across different contexts; e.g., the
rows may have different values if the expression is applied to different tables:

    >>> table = hl.utils.range_table(5, 1).annotate(x=hl.rand_bool(0.5, seed=0))
    >>> table.x.collect()  # doctest: +SKIP_OUTPUT_CHECK
    [0.5488135008937808,
     0.7151893652121089,
     0.6027633824638369,
     0.5448831893094143,
     0.42365480398481625]

    >>> table = hl.utils.range_table(5, 1).annotate(x=hl.rand_bool(0.5, seed=0))
    >>> table.x.collect()  # doctest: +SKIP_OUTPUT_CHECK
    [0.5488135008937808,
     0.7151893652121089,
     0.6027633824638369,
     0.5448831893094143,
     0.42365480398481625]

    >>> table = hl.utils.range_table(5, 5).annotate(x=hl.rand_bool(0.5, seed=0))
    >>> table.x.collect()  # doctest: +SKIP_OUTPUT_CHECK
    [0.5488135008937808,
     0.9595974306263271,
     0.42205690070893265,
     0.828743805759555,
     0.6414977904324134]

The seed can also be set globally using :func:`.set_global_seed`. This sets the
seed globally for all subsequent Hail operations, and a pipeline will be
guaranteed to have the same results if the global seed is set right beforehand:

    >>> hl.set_global_seed(0)
    >>> hl.eval(hl.array([hl.rand_unif(0, 1), hl.rand_unif(0, 1)]))  # doctest: +SKIP_OUTPUT_CHECK
    [0.6830630912401323, 0.4035978197966855]

    >>> hl.set_global_seed(0)
    >>> hl.eval(hl.array([hl.rand_unif(0, 1), hl.rand_unif(0, 1)]))  # doctest: +SKIP_OUTPUT_CHECK
    [0.6830630912401323, 0.4035978197966855]


.. autosummary::

    rand_bool
    rand_beta
    rand_cat
    rand_dirichlet
    rand_gamma
    rand_norm
    rand_pois
    rand_unif
    shuffle


.. autofunction:: rand_bool
.. autofunction:: rand_beta
.. autofunction:: rand_cat
.. autofunction:: rand_dirichlet
.. autofunction:: rand_gamma
.. autofunction:: rand_norm
.. autofunction:: rand_pois
.. autofunction:: rand_unif
.. autofunction:: shuffle
