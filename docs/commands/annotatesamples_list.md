<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

### Examples

<h4 class="example">Importing a list of samples into sample annotations</h4>

We have a file with a list of samples in one batch:
```
$ cat ~/Batch1_samples.txt
PT-1234 
PT-1235 
PT-1236 
PT-1237 
PT-1238 
PT-1239 
```

To annotate from this file, we must merely specify where to put it in sample annotations.

```
$ hail [read / import / previous commands] \
    annotatesamples list \
        -i file:///user/me/samples.tsv \
        -r sa.batch1
```

   This will read the file and produce annotations of the following schema:

```
Sample annotations:
sa: Struct { 
    batch1: Boolean
}
```



<h4 class="example">Importing a list of samples from a file with multiple columns and a non-default way of encoding missingness</h4>

```
$ cat ~/samples2.tsv
Batch   PT-ID
1kg     PT-0001
1kg     PT-0002
study1  PT-0003
study3  PT-0003
.       PT-0004
1kg     PT-0005
.       PT-0006
1kg     PT-0007
```

This file does not have non-string types, but it does have a sample column identifier that is not "Sample", and missingness encoded by "." instead of "NA".  The command line should read:

```
$ hail [read / import, previous commands] \
    annotatesamples table \
        -i file:///user/me/samples2.tsv \
        --sampleheader PT-ID \
        --missing "." \
        --root sa.group1.batch
```

</div>