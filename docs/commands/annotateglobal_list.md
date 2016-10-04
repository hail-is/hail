<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Importing a gene list into the global annotations</h4>
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
    read file.vds \ 
    annotateglobal list -i /tmp/genes.txt -r global.genes \
    printschema --global \
    showglobals
    
          
hail: info: running: read ../data/profile.vds/   
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


<h4 class="example>Filter variants from a user-defined gene list</h4>
```
$ hail 
    read file.vds \ 
    annotateglobal list -i /tmp/genes.txt -r global.genes --as-set \
    filtervariants expr --keep -c 'global.genes.contains(va.gene)' \
    write -o genes_of_interest.vds
```
</div>