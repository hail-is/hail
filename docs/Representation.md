
# <a name="Representation"></a> Data Representation

```
                Columns keyed
                by Samples  -->
            __________________________    __
           |                          |  |  |
           |                          |  |  |
   Rows    |    __                    |  |  |   Variant
 keyed by  |   |__| Genotype          |  |  | Annotations
 Variants  |                          |  |  |
           |                          |  |  |
    |      |                          |  |  |
    |      |                          |  |  |
    V      |                          |  |  |
           |                          |  |  |
           |                          |  |  |
           |                          |  |  |
           |__________________________|  |__|
            __________________________
           |__________________________|
                Sample annotations
           
                     ____
                    |    | Global annotations
                    |____|           
```

The above cartoon depicts the rough organization of the data stored in Hail.  The majority of the data is genotype information, which can be thought of as a matrix with columns keyed by [**sample**](#sample) objects, and rows keyed by [**variant**](#variant) objects.  Each cell of the matrix is a [**genotype**](#genotype) object.

For more information about **annotations**, [see the documentation here](#Annotations)

Hail's internal representation for genotypes is conceptually similar to VCF.  Hail only supports diploid genotypes and does not yet store phasing information.  Hail uses a fixed set of genotype fields corresponding to a VCF format field of the form:
```
GT:AD:DP:GQ:PL
```

In addition, Hail considers OD = DP - sum(AD).

Hail makes the following assumptions about the genotype fields:
 - if both are present, PL(GT) = 0
 - GQ is present if and only if PL is present
 - GQ is the difference of the two smallest PL entries
 - if OD is present, then AD is present
 - if they are all present, sum(AD) + OD = DP
 - sum(AD) <= DP

Internally, Hail preserves these invariants.  On import, Hail filters (sets to no-call) any genotype that violates these invariants and prints a warning message.
