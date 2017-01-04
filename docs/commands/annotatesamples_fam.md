<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Annotation Schema
The annotation names, types, and missing values are shown below, assuming the default root `sa.fam`.

Field | Annotation | Type | Missing
---|---|---|---
Family ID | `sa.fam.famID` | String | `0`
Sample ID | `s` | String | |
Paternal ID | `sa.fam.patID` | String | `0`
Maternal ID | `sa.fam.matID` | String | `0`
Sex | `sa.fam.isFemale` | Boolean | `NA`, `-9`, `0`
Case-control phenotype | `sa.fam.isCase` | Boolean | `0`, `-9`, non-numeric, and `-m` arg if given
Quantitative phenotype | `sa.fam.qPheno` |Double |  either `NA` or -m arg if given

</div>

<div class="cmdsubsection">
### Examples

<h4>Import data from a tab-separated PLINK .fam file into sample annotations with a case-control phenotype</h4>

```
annotatesamples fam -i myStudy.fam
```

<h4>Import data from a PLINK .fam file into sample annotations with a quantitative phenotype</h4>

In Hail, unlike Plink, the user must be explicit about whether the phenotype is case-control or quantitative. Importing a quantitive phenotype without the `-q` flag will return an error (unless all values happen to be `0`, `1`, `2`, and `-9`).

```
annotatesamples fam -i myStudy.fam -q
```

<h4>Import data from a whitespace-delimited PLINK .fam file into sample annotations with a case-control phenotype</h4>

```
annotatesamples fam -i myStudy.fam -d "\\s+"
```

</div>

