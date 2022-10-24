.. _sec-random-functions:

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

    >>> hl.eval(x)
    0.9828239225846387

    >>> hl.eval(x)
    0.9828239225846387

    >>> hl.eval(hl.rand_unif(0, 1))
    0.49094525115847415

    >>> hl.eval(hl.rand_unif(0, 1))
    0.3972543766997359

    >>> hl.eval(hl.array([x, x, x]))
    [0.9828239225846387, 0.9828239225846387, 0.9828239225846387]

If the three values in the last expression should be distinct, three separate
calls to :func:`.rand_unif` should be made:

    >>> a = hl.rand_unif(0, 1)
    >>> b = hl.rand_unif(0, 1)
    >>> c = hl.rand_unif(0, 1)
    >>> hl.eval(hl.array([a, b, c]))
    [0.992090957001768, 0.9564448098124774, 0.3905029525642664]

Within the rows of a :class:`.Table`, the same expression will yield a
consistent value within each row, but different (random) values across rows:

    >>> table = hl.utils.range_table(5, 1)
    >>> table = table.annotate(x1=x, x2=x, rand=hl.rand_unif(0, 1))
    >>> table.show()
    +-------+----------+----------+----------+
    |   idx |       x1 |       x2 |     rand |
    +-------+----------+----------+----------+
    | int32 |  float64 |  float64 |  float64 |
    +-------+----------+----------+----------+
    |     0 | 4.68e-01 | 4.68e-01 | 6.36e-01 |
    |     1 | 8.24e-01 | 8.24e-01 | 9.72e-01 |
    |     2 | 7.33e-01 | 7.33e-01 | 1.43e-01 |
    |     3 | 8.99e-01 | 8.99e-01 | 5.52e-01 |
    |     4 | 4.03e-01 | 4.03e-01 | 3.50e-01 |
    +-------+----------+----------+----------+


The same is true of the rows, columns, and entries of a :class:`.MatrixTable`.

Setting a seed
==============

All random functions can take a specified seed as an argument. This guarantees
that multiple invocations of the same function within the same context will
return the same result, e.g.

    >>> hl.eval(hl.rand_unif(0, 1, seed=0))
    0.2664972565962568

    >>> hl.eval(hl.rand_unif(0, 1, seed=0))
    0.2664972565962568

    >>> table = hl.utils.range_table(5, 1).annotate(x=hl.rand_unif(0, 1, seed=0))
    >>> table.x.collect()
    [0.5820244750020055,
     0.33150686392731943,
     0.20526631289173847,
     0.6964416913998893,
     0.6092952493383876]

    >>> table = hl.utils.range_table(5, 5).annotate(x=hl.rand_unif(0, 1, seed=0))
    >>> table.x.collect()
    [0.5820244750020055,
     0.33150686392731943,
     0.20526631289173847,
     0.6964416913998893,
     0.6092952493383876]

However, moving it to a sufficiently different context will produce different
results:

    >>> table = hl.utils.range_table(7, 1)
    >>> table = table.filter(table.idx >= 2).annotate(x=hl.rand_unif(0, 1, seed=0))
    >>> table.x.collect()
    [0.20526631289173847,
     0.6964416913998893,
     0.6092952493383876,
     0.6404026938964441,
     0.5550464170615771]

In fact, in this case we are getting the tail of

    >>> table = hl.utils.range_table(7, 1).annotate(x=hl.rand_unif(0, 1, seed=0))
    >>> table.x.collect()
    [0.5820244750020055,
     0.33150686392731943,
     0.20526631289173847,
     0.6964416913998893,
     0.6092952493383876,
     0.6404026938964441,
     0.5550464170615771]

Reproducibility across sessions
===============================

The values of a random function are fully determined by three things:

* The seed set on the function itself. If not specified, these are simply
  generated sequentially.
* Some data uniquely identifying the current position within a larger context,
  e.g. Table, MatrixTable, or array. For instance, in a :func:`.range_table`,
  this data is simply the row id, as suggested by the previous examples.
* The global seed. This is fixed for the entire session, and can only be set
  using the ``global_seed`` argument to :func:`.init`.

To ensure reproducibility within a single hail session, it suffices to either
manually set the seed on every random function call, or to call
:func:`.reset_global_randomness` at the start of a pipeline, which resets the
counter used to generate seeds.

    >>> hl.reset_global_randomness()
    >>> hl.eval(hl.array([hl.rand_unif(0, 1), hl.rand_unif(0, 1)]))
    [0.9828239225846387, 0.49094525115847415]

    >>> hl.reset_global_randomness()
    >>> hl.eval(hl.array([hl.rand_unif(0, 1), hl.rand_unif(0, 1)]))
    [0.9828239225846387, 0.49094525115847415]

To ensure reproducibility across sessions, one must in addition specify the
`global_seed` in :func:`.init`. If not specified, the global seed is chosen
randomly. All documentation examples were computed using ``global_seed=0``.

    >>> hl.stop()                                                   # doctest: +SKIP
    >>> hl.init(global_seed=0)                                      # doctest: +SKIP
    >>> hl.eval(hl.array([hl.rand_unif(0, 1), hl.rand_unif(0, 1)])) # doctest: +SKIP
    [0.9828239225846387, 0.49094525115847415]

.. autosummary::

    rand_bool
    rand_beta
    rand_cat
    rand_dirichlet
    rand_gamma
    rand_norm
    rand_pois
    rand_unif
    rand_int32
    rand_int64
    shuffle


.. autofunction:: rand_bool
.. autofunction:: rand_beta
.. autofunction:: rand_cat
.. autofunction:: rand_dirichlet
.. autofunction:: rand_gamma
.. autofunction:: rand_norm
.. autofunction:: rand_pois
.. autofunction:: rand_unif
.. autofunction:: rand_int32
.. autofunction:: rand_int64
.. autofunction:: shuffle
