<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes

The input file specifies the the new sample names.  It is a two
column, tab-separated file.  The first column is the current sample
name, the second column is the new sample name.  Samples which do not
appear in the input will not be renamed.  Lines in the input which
don't correspond to any sample in the current dataset will be ignored.

`exportsamples` can be used to generate a template for renaming
samples.  For example, suppose you want to rename samples to remove
spaces.  First, run:

```
$ hail read -i /path/to/my.vds exportsamples -c 's.id, s.id' -o sample.map
```

Then edit `sample.map` to remove spaces from the sample names in the
second column.  Then run your desired analysis in Hail:

```
$ hail read -i /path/to/my.vds renamesamples -i  sample.map ...
```

Renaming samples is fast and there is no need to resave the dataset
before performing analyses.

</div>