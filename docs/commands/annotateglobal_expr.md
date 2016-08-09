<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes

The input expression to `annotateglobal expr` as specified by the `-c` flag looks something like the following:
```
annotateglobal expr -c 'global.first10genes = global.genes[:10]' 
```

To break down this expression:
```
annotateglobal expr \
   -c   "global.first10genes                =         global.genes[:10]     [ , ... ]"
                  ^                         ^                ^                  ^
         *period-delimited path         equals sign       expression      (optional) comma 
       starting with 'global'          delimits path                      followed by more 
       where the annotation will      and expression                    annotation statements
             be placed
```

If an annotation path contains special characters (like whitespace, `:`, `-`, etc...), access it by escaping an identifier with backticks: 
```
global.custom.`this field has whitespace`
global.`identifier...with...dots`.annotation1
global.`layer one`.`layer two`.`layer three`
```

</div>

<div class="cmdsubsection">
### Accessible fields

Identifier | Description
:-: | ---
`global` | Global annotations
`variants` | variants and their annotations, an [aggregable](#aggregables)
`samples` | samples and their annotations, an [aggregable](#aggregables)

**Namespace of `samples` aggregable:**

Identifier | Description
:-: | ---
`global` | Global annotations
`s` | Sample
`sa` | Sample annotations

**Namespace of `variants` aggregable:**

Identifier | Description
:-: | ---
`global` | Global annotations
`v` | Variant
`va` | Variant annotations

</div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Count the total number of cases and controls in the dataset</h4>
```
annotateglobal expr -c 'global.nCase = samples.count(sa.pheno.isCase), 
                        global.nControl = samples.count(!sa.pheno.isCase),
                        global.nSample = samples.count(true)'
showglobals

hail: info: running: showglobals
Global annotations: `global' = {
  "nCase" : 215,
  "nIndel" : 444,
  "nVar" : 619
}
```


<h4 class="example">Count the total number of SNPs and Indels in the dataset</h4>
```
annotateglobal expr -c 'global.nSNP = variants.count(v.altAllele.isSNP), 
                        global.nIndel = variants.count(v.altAllele.isIndel),
                        global.nVar = variants.count(true)'
showglobals

hail: info: running: showglobals
Global annotations: `global' = {
  "nSNP" : 590,
  "nIndel" : 585,
  "nVar" : 1175
}
```


<h4 class="example">Compute summary statistics for the number of singletons per sample</h4>
```
sampleqc
annotateglobal expr -c 'global.singletonStats = samples.stats(sa.qc.nSingleton)'
showglobals

hail: info: running: showglobals
Global annotations: `global' = {
  "singletonStats" : {
    "mean" : 105.2,
    "stdev" : 20.5,
    "min" : 0,
    "max" : 410,
    "nNotMissing" : 812,
    "sum" : 81622
  }
}
```
</div>