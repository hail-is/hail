# `annotateglobal list`

This module is a subcommand of `annotateglobal`, and loads a text file as an `Array[String]` or `Set[String]` in the global annotations.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--input <file>` | `-i` | **Required** | Path to file
`--root <root>` | `-r` | **Required** | Annotation path: period-delimited path starting with `global`
`--as-set` | `--` | `false` | Load the file as a `Set[String]` instead of an `Array[String]`

____

#### Examples

Putting a list into the global table:
```
$ cat /tmp/genes.txt
SCN2A
SONIC-HEDGEHOG
PRNP
ALDH4A1
LEP
OSM
TSC1
TSC2

$ hail 
    read -i file.vds \ 
    annotateglobal list -i /tmp/genes.txt -r global.genes \
    printschema --global \
    showglobals
    
          
hail: info: running: read -i ../data/profile.vds/   
hail: info: running: annotateglobal list -i /tmp/genes.txt -r global.genes
hail: info: running: printschema --global    
   Global annotation schema:
   global: Struct {
       genes: Array[String]
   }
hail: info: running: showglobals
   Global annotations: `global' = {
     "genes" : [ "SCN2A", "SONIC-HEDGEHOG", "PRNP", "ALDH4A1", "LEP", "OSM", "TSC1", "TSC2" ]
   }
```


Using `annotateglobal list` to filter on gene:
```
$ hail 
    read -i file.vds \ 
    annotateglobal list -i /tmp/genes.txt -r global.genes --as-set \
    filtervariants expr --keep -c 'global.genes.contains(va.gene)' \
    write -o genes_of_interest.vds
```