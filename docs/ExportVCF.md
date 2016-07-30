# Exporting to VCF

Hail contains an `exportvcf` module which will write out the internal VDS in a vcf file to the [VCF 4.2 spec](https://samtools.github.io/hts-specs/VCFv4.2.pdf).

Command line arguments:
 - `-a <file>` -- File to append to VCF header.  Optional.
 - `-o <output file> | --output <output file>` -- Output file.  Required.

Example `exportvcf` commands:
```
hail read -i /path/to/file.vds exportvcf -o /path/to/file.vcf
```

```
hail read -i /path/to/file.vds exportvcf -o /path/to/file.vcf.bgz
```

## What Hail writes out

Importing a VCF into Hail's VDS then exporting it as a VCF will not result in exactly the same VCF.  The differences, however, are limited to the header and INFO field.  Hail's VCF header will contain:
 - FORMAT lines, with the format "GT:AD:DP:GQ:PL"
 - FILTER lines, if present in original imported VCF
 - INFO lines (see exported annotations below)
 
Hail's VCF header will not contain:
 - contig lines
 - lines added by external tools (bcftools, GATK, etc.)

<a name="annotations"></a>
## Annotations included in the info field

In order to determine what to export in the INFO field of a VCF, Hail looks in the variant annotation schema under `va.info`.  The workflow `hail importvcf <args> exportvcf <args>` will write out the same info field that was read in.  This means that problems can emerge when a workflow becomes more complicated.  If samples or genotypes are filtered after importing a VCF, the value stored in `va.info.AC` value may no longer reflect the number of called alternate alleles in the variant dataset.  If this state is exported to VCF, downstream tools may produce false results.

The solution to this problem is to use `annotatevariants` to create new annotations in `va.info` (or overwrite existing annotations).  For example, in order to produce an accurate "AC" field, one should run `variantqc` and copy the `va.qc.AC` field:

```
[previous commands] annotatevariants -c 'va.info.AC = va.qc.AC' 
```

In the context of a larger workflow:

```
$ hail importvcf file.vcf.bgz \
    filtergenotypes --keep -c 'g.gq > 20' \
    sampleqc
    filtersamples --remove -c 'sa.qc.callRate < 0.99'
    variantqc
    annotatevariants -c 'va.info.AC = va.qc.AC'
    exportvcf -o fileQC.vcf.bgz
```

For more information about what types of functionality is available to copy and create annotations, [see the documentation here](ProgrammaticAnnotation.md)

**Hail's VCF will not contain any added sample annotations, or variant annotations not in `va.info`.**
