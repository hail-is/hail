<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

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

If an annotation path contains special characters (like whitespace, `:`, `-`, etc...), access it by escaping an identifier with backticks: 
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
`gs` | Genotype row [aggregable](#aggregables)

</div>

<div class="cmdsubsection">
### Examples:

<h4 class="example">Compute GQ statistics about heterozygotes, per variant</h4>
```
annotatevariants expr -c 'va.gqHetStats = gs.filter(g => g.isHet).map(g => g.gq).stats()
exportvariants -o out.txt -c 'variant = v, het_gq_mean = va.gqHetStats.mean'
```

<h4 class="example">Collect a list of sample IDs with non-ref calls in LOF variants</h4>
```
filtervariants expr --keep -c 'va.consequence == "LOF"'
annotatevariants expr -c 'va.nonRefSamples = gs.filter(g => g.isCalledNonRef).map(g => s.id).collect()'
```
</div>