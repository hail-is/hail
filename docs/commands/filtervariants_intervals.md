<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

### Notes:
Hail expects an interval file to contain either three or five fields per line in the following formats: 

 - `contig:start-end`
 - `contig  start  end` (tab-separated)
 - `contig  start  end  direction  target` (tab-separated)
 
In either case, Hail will use only the `contig`, `start`, and `end` fields.  Each variant is evaluated against each line in the interval file, and any match will mark the variant to be kept / excluded based on the presence of the `--keep` and `--remove` flags.  _Note: "start" and "end" match positions inclusively, e.g. start <= position <= end_

```
$ hail read file.vds
    filtervariants intervals -i intervals.txt --keep
    ...
```

</div>