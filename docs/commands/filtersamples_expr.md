<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

### Description:

Use the Hail expression language to supply a boolean expression involving the following exposed data structures:

Exposed Name | Description
:-: | ---
 `s`  | sample
 `sa` | sample annotation
 `global` | global annotation
 `gs` | genotype column [aggregable](reference.html#aggregables)

   
For more information about these exposed objects and how to use them, see the documentation on [representation](reference.html#Representation) and the [Hail expression language](reference.html#HailExpressionLanguage).
   
```
$ hail read file.vds
    filtersamples expr -c 'sa.qc.callRate > 0.95' --keep
    ...
```


</div>