# `annotateglobal table`

This module is a subcommand of `annotateglobal`, and loads a text file by column as an `Array[Struct]` in global annotations.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--input <file>` | `-i` | **Required** | Path to file
`--root <root>` | `-r` | **Required** | Annotation path: period-delimited path starting with `global`
`--types <type-mapping>` | `-t` | All cols `String` | specify data types of fields, in a comma-delimited string of `name: Type` elements
`--missing <missing-value>` | `-m` | `NA` | Indicate the missing value in the table, if not "NA"
`--delimiter <regex>` | `-d` | `\t` | Indicate the field delimiter
`--comment <comment-char>` | `--` | **None** | Skip lines starting with the given pattern
`--no-header` | `--` | -- | Indicate that the file has no header.  Columns will instead be read as numbered, from `_0, _1, _2, ... _N`
`--impute` | `--` | -- | Turn on type imputation (works for Int, Double, Boolean, and Variant)

____

#### Example

Putting a table into the global annotations:
```
$ cat /tmp/file1.txt
GENE    PLI     EXAC_LOF_COUNT
Gene1   0.12312 2
Gene2   0.99123 0
Gene3   NA      NA
Gene4   0.9123  10
Gene5   0.0001  202

$ hail read -i ../data/profile.vds/ \
    annotateglobal table -i /tmp/file1.txt -r global.genes -t "PLI: Double, EXAC_LOF_COUNT: Int" \
    printschema --global \
    showglobals
    
    
hail: info: running: read -i ../data/profile.vds/
hail: info: running: annotateglobal table -i /tmp/file1.txt -r global.genes -t 'PLI: Double, EXAC_LOF_COUNT: Int'
hail: info: running: printschema --global
Global annotation schema:
global: Struct {
    genes: Array[Struct {
        GENE: String,
        PLI: Double,
        EXAC_LOF_COUNT: Int
    }]
}
hail: info: running: showglobals
Global annotations: `global' = {
  "genes" : [ {
    "GENE" : "Gene1",
    "PLI" : 0.12312,
    "EXAC_LOF_COUNT" : 2
  }, {
    "GENE" : "Gene2",
    "PLI" : 0.99123,
    "EXAC_LOF_COUNT" : 0
  }, {
    "GENE" : "Gene3",
    "PLI" : null,
    "EXAC_LOF_COUNT" : null
  }, {
    "GENE" : "Gene4",
    "PLI" : 0.9123,
    "EXAC_LOF_COUNT" : 10
  }, {
    "GENE" : "Gene5",
    "PLI" : 1.0E-4,
    "EXAC_LOF_COUNT" : 202
  } ]
}
```