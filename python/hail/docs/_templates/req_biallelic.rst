Requires the dataset to contain no multiallelic variants (``vds.was_split == True``).
Use :py:meth:`~hail.VariantDataset.split_multi` to split multiallelic sites
or :py:meth:`~hail.VariantDataset.filter_multi` to remove them.