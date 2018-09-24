Annotation
==========

Annotations are Hail's way of adding data fields to Hail's tables and matrix
tables.

Create a nested annotation
--------------------------

:**description**: Add a new field ``gq_mean`` as a nested field inside ``info``

:**code**:

    >>> mt = mt.annotate_rows(info=mt.info.annotate(gq_mean=hl.agg.mean(mt.GQ)))

:**dependencies**: :meth:`.StructExpression.annotate`, :meth:`.MatrixTable.annotate_rows`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            To add a new field ``gq_mean`` as a nested field inside ``info``,
            instead of a top-level field, we need to annotate the ``info`` field itself.

            Construct an expression ``mt.info.annotate(gq_mean=...)`` which adds the field
            to ``info``. Then, reassign this expression to ``info`` using
            :meth:`.MatrixTable.annotate_rows`.

Remove a nested annotation
--------------------------

:**description**: Drop a field ``AF``, which is nested inside the ``info`` field.

To drop a nested field ``AF``, construct an expression ``mt.info.drop('AF')``
which drops the field from its parent field, ``info``. Then, reassign this
expression to ``info`` using :meth:`.MatrixTable.annotate_rows`.

:**code**:

    >>> mt = mt.annotate_rows(info=mt.info.drop('AF'))

:**dependencies**: :meth:`.StructExpression.drop`, :meth:`.MatrixTable.annotate_rows`