# Filtering

Hail includes three filtering modules:
 - `filtervariants`
 - `filtersamples`
 - `filtergenotypes`

The `filtervariants` module contains the following submodules:

- `intervals`: filter by an interval list [(skip to)](#vIntervals)
- `list`: filter by a variant list [(skip to)](#vList)
- `expr`: filter by Hail expressions [(skip to)](#vExpr)
- `all`: drop all variants [(skip to)](#vAll)

The `filtersamples` module contains the following submodules:

- `list`: filter by a sample list [(skip to)](#sList)
- `expr`: filter by Hail expressions [(skip to)](#sExpr)
- `all`: drop all samples [(skip to)](#sAll)

The `filtergenotypes` module filters solely on Hail expressions. [(skip to)](#genotypes)

____

### Submodules of `filtervariants`:

____

<a name="vIntervals"></a>
#### `filtervariants intervals`

Usage:

 - `-i | --input <file>` -- path to interval list file
 - `--keep/--remove` -- keep or remove variants within an interval from the file

Hail expects an interval file to contain either three or five fields per line in the following formats: `contig:start-end` or `contig  start  end  direction  target` (tab-separated).  In either case, Hail will use only the `contig`, `start`, and `end` fields.  Each variant is evaluated against each line in the interval file, and any match will mark the variant to be kept / excluded based on the presence of the `--keep` and `--remove` flags.  _Note: "start" and "end" match positions inclusively, e.g. start <= position <= end_

```
$ hail read -i file.vds
    filtervariants intervals -i intervals.txt --keep
    ...
```

____

<a name="vList"></a>
#### `filtervariants list`

Usage:

 - `-i | --input <file>` -- path to variant list file
 - `--keep/--remove` -- keep or remove variants contained in the file

Hail expects a .variant_list file to contain a variant per in line following format: `contig:pos:ref:alt1,alt2,...,altN`.  Variants in the dataset will be kept / excluded based on the presence of the `--keep` and `--remove` flags.

```
$ hail read -i file.vds
    filtervariants list -i variants.txt --keep
    ...
```

____

<a name="vExpr"></a>
#### `filtervariants expr`

Usage:

 - `-c | --condition <file>` -- hail language expression
 - `--keep/--remove` -- keep or remove variants where the condition is true

Use the Hail expression language to supply a boolean expression involving the following exposed data structures:

Exposed Name | Description
:-: | ---
`v`  | variant
`va` | variant annotation
`gs` | genotype row [aggregable](HailExpressionLanguage.md#aggregables)

    
For more information about these exposed objects and how to use them, see the documentation on [representation](Representation.md) and the [Hail expression language](HailExpressionLanguage.md).

```
$ hail read -i file.vds
    filtervariants expr -c 'v.contig == "X"' --keep
    ...
```

____

<a name="vAll"></a>
#### `filtervariants all`

Removes all variants from VDS.

```
$ hail read -i file.vds
    filtervariants all
    ...
```

____

### Submodules of `filtersamples`:

____

<a name="sList"></a>
#### `filtersamples list`

Usage:

 - `-i | --input <file>` -- path to variant list file
 - `--keep/--remove` -- keep or remove samples contained in the file

Hail expects a sample list file to contain one sample per line, with no other fields.

```
$ hail read -i file.vds
    filtersamples list -i samples.txt --keep
    ...
```

____

<a name="sExpr"></a>
#### `filtersamples expr`

Usage:

 - `-c | --condition <file>` -- hail language expression
 - `--keep/--remove` -- keep or remove samples where the condition is true

Use the Hail expression language to supply a boolean expression involving the following exposed data structures:

Exposed Name | Description
:-: | ---
 `s`  | sample
 `sa` | sample annotation
 `gs` | genotype column [aggregable](HailExpressionLanguage.md#aggregables)

   
For more information about these exposed objects and how to use them, see the documentation on [representation](Representation.md) and the [Hail expression language](HailExpressionLanguage.md).
   
```
$ hail read -i file.vds
    filtersamples expr -c 'sa.qc.callRate > 0.95' --keep
    ...
```

____

<a name="sAll"></a>
#### `filtersamples all`

Removes all samples from VDS.  The variants and variant annotations will remain, making it a sites-only VDS.

```
$ hail read -i file.vds
    filtersamples all
    ...
```

____

<a name="genotypes"></a>
### `filtergenotypes`

The filter genotypes module has only one function, the `expr` function, so it is not broken into submodules.  

**Usage:**

 - `-c | --condition <file>` -- hail language expression
 - `--keep/--remove` -- keep or remove genotypes where the condition is true

In `filtergenotypes`, removed genotypes will be set to missing.  Use the Hail expression language to supply a boolean expression involving the following exposed data structures:

Exposed Name | Description
:-: | ---
 `g`  | genotype
 `s`  | sample
 `sa` | sample annotation
 `v`  | variant
 `va` | variant annotation
   
For more information about these exposed objects and how to use them, see the documentation on [representation](Representation.md) and the [Hail expression language](HailExpressionLanguage.md).
   
## Examples 

#### Using files

```
filtervariants intervals -c 'file.interval_list' --keep
```

```
filtervariants list -i 'file.variants' --keep
```

```
filtersamples list -i 'file.sample_list' --remove
```

#### Using expressions

```
filtervariants expr -c 'v.contig == "5"' --keep
```

```
filtervariants expr -c 'va.pass' --keep
```

```
filtervariants expr -c '!va.pass' --remove
```

```
[after importvcf & splitmulti]
filtervariants expr -c 'va.info.AC[va.aIndex] > 1' --keep 
```

```
filtergenotypes -c 'g.ad(1).toDouble / g.dp < 0.2' --remove
```

```
filtersamples expr -c 'if ("DUMMY" ~ s.id) 
    sa.qc.rTiTv > 0.45 && sa.qc.nSingleton < 10000000
    else 
    sa.qc.rTiTv > 0.40' --keep
```

```
filtergenotypes -c 'g.gq < 20 || (g.gq < 30 && va.info.FS > 30)' --remove
```

**Remember:**
 - All variables and values are case sensitive
 - Missing values will always be **excluded**, regardless of `--keep`/`--remove`.  Expressions in which any value is missing will evaluate to missing.
 
____ 
 
#### Dropping all samples / variants from a VDS

```
$ hail read -i with_genotypes.vds 
    filtersamples all 
    write -o sites_only.vds
```

```
$ hail read -i file.vds \
    filtervariants all
    write -o sample_info_only.vds
```
