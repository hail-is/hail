.. note::

    Requires the dataset to be keyed by two fields:

     - `locus` (type :class:`.tlocus`)
     - `alleles` (type :class:`.tarray` of :py:data:`.tstr`)

    Also requires that `locus` is the partition key.