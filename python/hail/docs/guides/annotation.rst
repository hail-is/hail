Annotation
==========

Annotations are Hail's way of adding data fields to Hail's tables and matrix
tables.

Create a nested annotation
--------------------------

To add a new field ``gq_mean`` as a nested field inside ``info``,
construct an expression ``mt.info.annotate(gq_mean=...)`` which adds the field
to ``info``. Then, reassign this expression to ``info`` using
:meth:`.MatrixTable.annotate_rows`.

>>> mt = mt.annotate_rows(info=mt.info.annotate(gq_mean=hl.agg.mean(mt.GQ)))

Remove a nested annotation
--------------------------

To drop a nested field ``AF``, construct an expression ``mt.info.drop('AF')``
which drops the field from its parent field, ``info``. Then, reassign this
expression to ``info`` using :meth:`.MatrixTable.annotate_rows`.

>>> mt = mt.annotate_rows(info=mt.info.drop('AF'))
