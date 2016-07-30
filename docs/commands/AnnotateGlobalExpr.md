# `annotateglobal expr`

This module is a subcommand of `annotateglobal`, and uses the hail expr language to compute new annotations from existing global annotations, as well as perform sample and variant aggregations.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--condition <expr>` | `-c` | **Required** | Hail annotation expression -- see below.

#### Description

These expressions look something like the following:
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

**Note**: if an annotation path contains special characters (like whitespace, `:`, `-`, etc...), access it by escaping an identifier with backticks: 
```
global.custom.`this field has whitespace`
global.`identifier...with...dots`.annotation1
global.`layer one`.`layer two`.`layer three`
```

#### Accessible fields

Identifier | Description
:-: | ---
`global` | Global annotations
`variants` | variants and their annotations, an [aggregable](../HailExpressionLanguage.md#aggregables)
`samples` | samples and their annotations, an [aggregable](../HailExpressionLanguage.md#aggregables)

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

____

#### Examples

Count the number of cases and controls.
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

____

Count the number of SNPs and Indels.
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

____

Compute statistics about number of singletons per sample.

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