<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

The `concordance` command computes the genotype call concordance between two bialellic datasets.  The concordance results are an annotation `Array[Array[Long]]`, which is a 5x5 table of counts with the following mapping:

```
  0: No Data (missing variant)
  1: No Call (missing genotype call)
  2: Hom Ref
  3: Heterozygous
  4: Hom Var
```
The first index is the left dataset, the second is the right.  For example, `concordance[3][2]` is the count of genotypes which were heterozygous on the left and hom ref on the right.

This command produces two new datasets and places them in the environment (you can access them with the `get` command).  

The first dataset, indicated by the --variants option, contains the concordance statistics per variant.  This dataset **contains no genotypes** (sites-only), and has only one variant annotation, `va.concordance`.  This is the concordance table for each variant in the outer join of the two datasets -- if the variant is present in only one dataset, all of the counts will lie in the axis `va.concordance[0][:]` (if it is missing on the left) or `va.concordance.map(x => x[0])` (if it is missing on the right).  This dataset also contains the global concordance statistics in `global.concordance`.

The second dataset, indicated by the --samples option, contains the concordance statistics per sample.  This dataset **contains no variants** (samples-only), and has only one sample annotation, `sa.concordance`.  This is a concordance table whose sum is the total number of variants in hte outer join of the two datasets.  The sum `sa[0].sum` is equal to the number of variants in the right dataset but not the left, and the sum `sa.concordance.map(x => x[0]).sum)` is equal to the number of variants in the left dataset but not the right.  This dataset also contains the global concordance statistics in `global.concordance`.

**Example:**

```
hail read RightDataset.vds
  put -n right
  read -i LeftDataset.vds
  concordance --right right --variants site_concordance --samples sample_concordance
  get -n site_concordance
  < aggregate/export site concordance metrics > 
  get -n sample_concordance
  < aggregate/export sample concordance metrics >
```

</div>