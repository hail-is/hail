This is the representation of and documentation for a Hail type. Values of this
type have two representations: the expression language representation and the
Python representation. This is confusing! There are two sets of documentation
for many objects like Variant and Genotype. In methods like
:py:meth:`~hail.VariantDataset.annotate_variants_expr` and 
:py:meth:`~hail.VariantDataset.filter_samples_expr`, Hail expressions are used
to manipulate Hail types to produce new annotations and filter samples, variants,
and genotypes programmatically. Some methods like 
:py:meth:`~hail.VariantDataset.query_variants` and 
:py:meth:`~hail.VariantDataset.query_samples` use expressions to slice, dice, and
reduce data so that it can fit in memory, and return the results as Python objects.
