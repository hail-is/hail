## <a name="exportingdata"></a> Exporting Data

#### When exporting the VDS to a VCF file, what file extension should I use?

.vcf.bgz

#### How do I create a sites-only VCF?

```
filtersamples all exportvcf -o /path/to/sitesonly.vcf.bgz
```

#### How do I substitute a custom string for the rsID field in the VCF exported by Hail?
 
```
annotatevariants expr -c 'va.rsid = va.contig + "_" + va.start + "_" + va.ref + "_" + va.alt'
exportvcf /path/to/myvcf.vcf.bgz
```