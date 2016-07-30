# `annotatesamples fam`

This module is a subcommand of `annotatesamples`, and loads Plink .fam file into sample annotations.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--input <file>` | `-i` | **Required** | Path to fam file
`--root <root>` | `-r` | **Required** | Annotation path root: period-delimited path starting with `sa`
`--quantitative` | `-q` | `false` | flag to indicate quantitative phenotype default is case-control
`--delimiter <sep>` | `-d` | `\t` | Indicate the field delimiter
`--missing <missing-val>` | `-m` | `NA` | Indicate the missing value in the table, if not "NA"

____

#### Description

The command

`annotatesamples fam -i myStudy.fam`

will add sample annotations for family ID, paternal ID, maternal ID, sex, and case-control phenotype, whereas

`annotatesamples fam -i myStudy.fam -q`

will interpret the phenotype as quantitative instead. The annotation names, types, and missing values are shown below, assuming the default root `sa.fam`.

Field | Annotation | Type | Missing
---|---|---|---
Family ID | `sa.fam.famID` | String | `0`
Sample ID | `s` | String |
Paternal ID | `sa.fam.patID` | String | `0`
Maternal ID | `sa.fam.matID` | String | `0`
Sex | `sa.fam.isFemale` | Boolean | `0`
Case-control phenotype | `sa.fam.isCase` | Boolean | `0`, `-9`, non-numeric, and `-m` arg if given
Quantitative phenotype | `sa.fam.qPheno` |Double |  either `NA` or -m arg if given

In Hail, unlike Plink, the user must be explicit about whether the phenotype is case-control or quantitative. Importing a quantitive phenotype without the `-q` flag will return an error (unless all values happen to be `0`, `1`, `2`, and `-9`).

If the .fam file is delimited by whitespace other than tabs (e.g., spaces), use delimiter parameter `\\s+`.
