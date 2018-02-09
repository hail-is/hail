.. note::

    Requires the dataset to be keyed by two fields:

     - `locus` (:class:`.TLocus`)
     - `alleles` (:class:`.TArray` of :class:`.TString`)

    Also requires that `locus` is the partition key.