<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes

The input expression to `annotateglobal exprbysample` as specified by the `-c` flag looks something like the following:
```
annotateglobal exprbysample -c 'global.first10genes = global.genes[:10]' 
```

To break down this expression:
```
annotateglobal exprbysample \
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
`samples` | samples and their annotations, an [aggregable](reference.html#aggregables)

**Namespace of `samples` aggregable:**

Identifier | Description
:-: | ---
`global` | Global annotations
`s` | Sample ID
`sa` | Sample annotations

</div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Count the total number of cases and controls in the dataset</h4>
```
annotateglobal exprbysample -c 'global.nCase = samples.filter(s => sa.pheno.isCase).count(), 
                                global.nControl = samples.filter(s => !sa.pheno.isCase).count(),
                                global.nSample = samples.count()'
showglobals

hail: info: running: showglobals
Global annotations: `global' = {
  "nCase" : 215,
  "nIndel" : 444,
  "nVar" : 619
}
```


<h4 class="example">Compute summary statistics for the number of singletons per sample</h4>
```
sampleqc
annotateglobal exprbysample -c 'global.singletonStats = samples.map(s => sa.qc.nSingleton).stats()
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