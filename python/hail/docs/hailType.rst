This is the representation of and documentation for a Hail type. Values of this
type have two representations: the expression language representation and the
Python representation. This is confusing! There are two sets of documentation
for many objects like Variant and Genotype. In methods like
`annotate_variants_expr` and `filter_samples_expr`, you will use Hail
expressions to manipulate Hail types, returning new datasets. In some methods
like `query_variants` and `query_samples`, you will use expressions to
manipulate Hail types, and return Python objects!
