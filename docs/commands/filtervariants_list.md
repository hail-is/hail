<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

### Description:

Hail expects a .variant_list file to contain a variant per in line following format: `contig:pos:ref:alt1,alt2,...,altN`.  Variants in the dataset will be kept / excluded based on the presence of the `--keep` and `--remove` flags.

```
$ hail read file.vds
    filtervariants list -i variants.txt --keep
    ...
```
</div>