.. note::

    Requires the dataset to have a compound row key:

     - `locus` (type :class:`.tlocus`)
     - `alleles` (type :class:`.tarray` of :py:data:`.tstr`)

    Also requires that `locus` is the partition key.