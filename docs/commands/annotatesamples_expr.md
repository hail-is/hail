<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Description

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
</div>

<div class="cmdsubsection">
### Accessible fields

Identifier | Description
:-: | ---
`s` | Sample
`sa` | Sample annotations
`global` | global annotation
`gs` | Genotype column [aggregable](reference.html#aggregables)
</div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Compute GQ statistics about heterozygotes, per sample</h4>
```
annotatesamples expr -c 'sa.gqHetStats = gs.filter(g => g.isHet).map(g => g.gq).stats()'
exportsamples -o out.txt -c 'sample = s, het_gq_mean = sa.gqHetStats.mean'
```

<h4 class="example">Collect a list of genes with the number of singleton LOF calls per sample</h4>
```
annotatevariants expr -c 'va.isSingleton = gs.map(g => g.nNonRefAlleles).sum() == 1'
annotatesamples expr -c 'sa.LOF_genes = gs.filter(g => va.isSingleton && g.isHet && va.consequence == "LOF").map(g => va.gene).collect()'
```
</div>