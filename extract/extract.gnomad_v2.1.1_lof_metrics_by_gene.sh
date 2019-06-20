
#!/bin/bash

wget ftp://console.cloud.google.com/storage/browser/gnomad-public/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz | 
zcat | 
bgzip -c > tofile

