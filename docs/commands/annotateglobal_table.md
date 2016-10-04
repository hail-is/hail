<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Example

<h4 class="example">Import data in table format into the global annotations</h4>
```
$ cat /tmp/file1.txt
GENE    PLI     EXAC_LOF_COUNT
Gene1   0.12312 2
Gene2   0.99123 0
Gene3   NA      NA
Gene4   0.9123  10
Gene5   0.0001  202

$ hail read ../data/profile.vds/ \
    annotateglobal table -i /tmp/file1.txt -r global.genes -t "PLI: Double, EXAC_LOF_COUNT: Int" \
    printschema --global \
    showglobals
    
    
hail: info: running: read ../data/profile.vds/
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
</div>