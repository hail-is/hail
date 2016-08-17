<div class="cmdhead"></div>

<div class="cmdsubsection">
Filter a user-defined set of alternate alleles for each variant. If all of a variant's alternate alleles are filtered, the variant itself is filtered. The condition expression is evaluated for each alternate allele. It is not evaluated for the reference (i.e. `aIndex` will never be zero).

If an allele is filtered, this method acts similarly to [`splitmulti`](#splitmulti). It transforms the genotype such that the filtered allele appears as a reference allele. The qualitative interpretation of this approach is a belief that the alternate is real but we want to shift the probability mass to 0 (thus changing our interpretation of 0 from "reference" to "reference or something not listed")

|Part|Description|Action|
|---|---|---|
|GT|the hard call|downcode the filtered alleles to reference|
|AD|allele depth|the filtered alleles' columns are eliminated and the value is added to the reference, e.g. filtering alleles 1 and 2 transforms `[25,5,10,20]` to `[40,20]`|
|DP|number of informative reads|no change|
|PL|Phred-likelihoods for each allele pair|downcode the filtered alleles and take the minimum of the likelihoods for each genotype|
|GQ|genotype quality|increasing-sort PL and take `PL[1] - PL[0]`|
</div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Expression Variables
The following table describes the variables in the scope of `COND_EXPR`

| Name | Description |
| --- | --- |
| `v` | variant |
| `va` | variant annotations |
| `aIndex` | allele index |

The following table describes the variables in the scope of `ANNO_EXPR`

| Name | Description |
| --- | --- |
| `v` | the _new_ variant |
| `va` | the _old_ variant annotations |
| `aIndices` | an array of the old allele indices (such that `aIndices[newIndex] = oldIndex` and `aIndices[0] = 0`) |

### Examples

#### A Conceptual Example

Consider three alleles.

```
GT: 1/2
GQ: 10

0 | 1000
1 | 1000   10
2 | 1000   0     20
  +-----------------
    0      1     2
```

Suppose we remove allele 2.

In the PL array, we convert occurences of 2 to 0 ("downcode to reference") and take minimums where there are multiple likelihoods for a single genotype; we also downcode GT to ref.

```
GT: 0/1
GQ: 10

0 | 20
1 | 0    10
  +-----------
    0    1
```

#### A Command Example

The following command removes alternate alleles whose allele count is zero and updates the alternate allele count annotation with the new indices.

```
... \
  filtersamples list --remove -i samples_to_exclude.txt \
  filteralleles --remove \
    -c 'va.info.AC[aIndex - 1] == 0' \
    -a 'va.info.AC = va.info.AC = aIndices[1:].map(i => va.info.AC[i - 1]),
      va.info.AF = aIndices[1:].map(i => va.info.AF[i - 1]),
      va.info.MLEAC = aIndices[1:].map(i => va.info.MLEAC[i - 1]),
      va.info.MLEAF = aIndices[1:].map(i => va.info.MLEAF[i - 1])'
```

Note that we must skip the first element of `aIndices` because it is mapping between the old and new *allele* indices, not the *alternate allele* indices.

</div>
