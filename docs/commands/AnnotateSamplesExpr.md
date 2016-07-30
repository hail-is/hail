# `annotatesamples expr`

This module is a subcommand of `annotatesamples`, and uses the hail expr language to compute new annotations from existing sample annotations, as well as perform genotype aggregation.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--condition <expr>` | `-c` | **Required** | Hail annotation expression -- see below.

#### Description

These expressions look something like the following:
```
annotatesamples expr -c 'sa.cohort1Male = sa.cohort == "Cohort1" && !sa.fam.isFemale' 
```

To break down this expression:
```
annotatesamples expr \
   -c   "sa.cohort1Male                     =       sa.cohort == "Cohort1" && !sa.fam.isFemale     [ , ... ]"
                  ^                         ^                           ^                            ^
         *period-delimited path         equals sign                  expression                (optional) comma 
       starting with 'sa'              delimits path                                           followed by more 
       where the annotation will      and expression                                         annotation statements
             be placed
```

**Note**: if an annotation path contains special characters (like whitespace, `:`, `-`, etc...), access it by escaping an identifier with backticks: 
```
sa.custom.`this field has whitespace`
sa.`identifier...with...dots`.annotation1
sa.`layer one`.`layer two`.`layer three`
```

#### Accessible fields

Identifier | Description
:-: | ---
`s` | Sample
`sa` | Sample annotations
`global` | global annotation
`gs` | Genotype column [aggregable](../HailExpressionLanguage.md#aggregables)

____

#### Examples

Compute GQ statistics about heterozygotes, per sample
```
annotatesamples expr -c 'sa.gqHetStats = gs.statsif(g.isHet, g.gq)'
exportsamples -o out.txt -c 'sample = s, het_gq_mean = sa.gqHetStats.mean'
```

____

Collect a list of genes with singleton LOF calls per sample
```
annotatevariants expr -c 'va.isSingleton = gs.stats(g.nNonRefAlleles).sum == 1'
annotatesamples expr -c 'sa.LOF_genes = gs.collect(va.isSingleton && g.isHet && va.consequence == "LOF", va.gene)'
```
