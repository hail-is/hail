.. note::

    The columnar format is *experimental*. Its 0.x versions make no
    compatibility guarantee: a later version of Hail may not read what this one
    writes, so do not use it for data you need to keep. Choosing it emits a
    warning. It stores data as Parquet files described by a JSON manifest, so
    tools other than Hail can read it. The `columnar format specification
    <https://github.com/hail-is/hail/blob/main/dev-docs/hail-query/columnar-format-spec.md>`__
    defines it.
