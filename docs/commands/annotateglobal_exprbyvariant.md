<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes

The input expression to `annotateglobal exprbyvariant` as specified by the `-c` flag looks something like the following:
```
annotateglobal expr -c 'global.first10genes = global.genes[:10]' 
```

To break down this expression:
```
annotateglobal exprbyvariant \
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
`variants` | variants and their annotations, an [aggregable](reference.html#aggregables)

**Namespace of `samples` aggregable:**

**Namespace of `variants` aggregable:**

Identifier | Description
:-: | ---
`global` | Global annotations
`v` | Variant
`va` | Variant annotations

</div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Count the total number of SNPs and Indels in the dataset</h4>
```
annotateglobal exprbyvariant -c 'global.nSNP = variants.filter(v => v.altAllele.isSNP).count(), 
                                 global.nIndel = variants.filter(v => v.altAllele.isIndel).count(),
                                 global.nVar = variants.count()'
showglobals

hail: info: running: showglobals
Global annotations: `global' = {
  "nSNP" : 590,
  "nIndel" : 585,
  "nVar" : 1175
}
```
</div>