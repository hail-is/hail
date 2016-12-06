<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

This command runs similarly to [annotatevariants expr](commands.html#annotatevariants_expr) but it dynamically splits multi-allelic sites,
 computes each expressions on each split allele separately, and returns an **Array** with one entry per
   non-ref allele for each expression.

During the evaluation of the expressions, two additional variant annotations are accessible:
1. `va.aIndex`: An **Int** indicating the index of the non-reference allele being evaluated
(amongst all alleles including the reference allele, so cannot be `0` )
2. `va.wasSplit`: A **Boolean** indicating whether this allele belongs to a multi-allelic site or not.

**Important Note**: When the alleles are split, the genotypes are downcoded and each non-reference allele
is represented using its minimal representation (see [splitmulti](commands.html#splitmulti) for more details).

Command line options:
 - `--propagate-gq` -- Propagate GQ instead of computing from PL when splitting alleles.

#### Accessible fields

Identifier | Description
:-: | ---
`v` | Variant
`va` | Variant annotations
`global` | global annotation
`gs` | Genotype row [aggregable](reference.html#aggregables)

</div>

<div class="cmdsubsection">
### Example:

<h4 class="example">Compute the number of samples carrying each non-reference allele</h4>
```
annotatealleles expr -c 'va.nNonRefSamples = gs.filter(g => g.isCalledNonRef).count()'
```

This expression produces a variant annotation `va.nNonRefSamples` of type **Array[Int]**,
where the nth entry of the array if the count of the number of samples carrying the nth non-reference allele.


</div>
