# Filtering

Hail includes three filtering modules:
 - `filtervariants`
 - `filtersamples`
 - `filtergenotypes`

The `filtervariants` module contains the following submodules:

- `intervals`: filter by an interval list [(skip to)](#filtervariants_intervals)
- `list`: filter by a variant list [(skip to)](#filtervariants_list)
- `expr`: filter by Hail expressions [(skip to)](#filtervariants_expr)
- `all`: drop all variants [(skip to)](#filtervariants_all)

The `filtersamples` module contains the following submodules:

- `list`: filter by a sample list [(skip to)](#filtersamples_list)
- `expr`: filter by Hail expressions [(skip to)](#filtersamples_expr)
- `all`: drop all samples [(skip to)](#filtersamples_all)

The `filtergenotypes` module filters solely on Hail expressions. [(skip to)](#filtergenotypes)

### Examples: 

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
filtervariants expr -c 'va.info.AC[va.aIndex - 1] > 1' --keep 
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
 
 
#### Dropping all samples / variants from a VDS

```
$ hail read -i with_genotypes.vds 
    filtersamples all 
    write -o sites_only.vds
```

```
$ hail read -i file.vds \
    filtervariants all
    write -o variant_info_only.vds
```
