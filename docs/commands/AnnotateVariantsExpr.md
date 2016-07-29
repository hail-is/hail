# `annotatevariants expr`

This module is a subcommand of `annotatevariants`, and uses the hail expr language to compute new annotations from existing variant annotations, as well as perform genotype aggregation.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--condition <expr>` | `-c` | **Required** | Hail annotation expression -- see below.

#### Description

These expressions look something like the following:
```
annotatevariants expr -c 'va.isSingleton = va.info.AC[va.aIndex] == 1'
```

To break down this expression:
```
annotatevariants expr \
   -c   "'va.isSingleton                    =           va.info.AC[va.aIndex] == 1'       [ , ... ]"
                  ^                         ^                     ^                           ^
         *period-delimited path         equals sign            expression             (optional) comma 
       starting with 'va'              delimits path                                  followed by more 
       where the annotation will      and expression                                annotation statements
             be placed
```

**Note**: if an annotation path contains special characters (like whitespace, `:`, `-`, etc...), access it by escaping an identifier with backticks: 
```
va.custom.`this field has whitespace`
va.`identifier...with...dots`.annotation1
va.`layer one`.`layer two`.`layer three`
```

#### Accessible fields

Identifier | Description
:-: | ---
`v` | Variant
`va` | Variant annotations
`global` | global annotation
`gs` | Genotype row [aggregable](../HailExpressionLanguage.md#aggregables)

____

#### Examples

Compute GQ statistics about heterozygotes, per variant
```
annotatevariants expr -c 'va.gqHetStats = gs.statsif(g.isHet, g.gq)'
exportvariants -o out.txt -c 'variant = v, het_gq_mean = va.gqHetStats.mean'
```

____

Collect a list of sample IDs with non-ref calls in LOF variants:
```
filtervariants expr --keep -c 'va.consequence == "LOF"'
annotatevariants expr -c 'va.nonRefSamples = gs.collect(g.isCalledNonRef, s.id)'
```
