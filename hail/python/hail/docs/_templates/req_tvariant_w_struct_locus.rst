.. note::

    Requires the dataset to be keyed by two fields:

     - `locus` (type :class:`.tlocus` **OR** type :class:`.tstruct` with two fields: `contig` with type :py:data:`.tstr` and `position` with type :py:data:`.tint32`)
     - `alleles` (type :class:`.tarray` of :py:data:`.tstr`)
