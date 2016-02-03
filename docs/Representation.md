# Representation

Hail's internal representation for variants and gentypes is
conceptually similar to VCF.  Hail only supports diploid genotypes and
does not yet store phasing information.  Hail uses a fixed set of
genotype fields corresponding to a VCF format field of the form:
```
GT:AD:DP:GQ:PL
```

In addition, Hail considers OD = DP - sum(AD).

Hail makes the following assumptions about the genotype fields:
 - if both are present, PL(GT) = 0
 - GQ is the difference of the smallest two PL entries
 - sum(AD) <= DP
 - if they are all present, sum(AD) + OD = DP
 - if OD is present, then AD is present (*)

Internally, Hail preserves these invariants.  On import, Hail filters
(sets to no-call) any genotype that violates these invariants and
prints a warning message.

(*) This is needlessly strict and maybe relaxed in the future.
