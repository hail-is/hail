# Filtering

Hail includes three filtering modules:
 - `filtervariants`
 - `filtersamples`
 - `filtergenotypes`
  
The modules share much of their command-line interface, but there are some important differences.  Hail's modern filtering system is distinguished by the user's ability to evaluate a scala expression for each variant, sample, or genotype to determine whether to keep or remove those data.  This system is incredibly powerful, and allows for filtering procedures that might otherwise take multiple iterations or even multiple tools to be completed in one command.

Command line arguments: 
 - `-c | --condition <cond>` -- filter expression (see below) or path to file of appropriate type
 - `--keep/--remove` -- determines behavior of file interval/list or boolean expression
  
## Using inclusion/exclusion files

1. `filtervariants` -- ".interval_list" file
 - Hail expects a .interval_list file to contain either three or five fields per line in the following formats: `contig:start-end` or `contig  start  end  direction  target` (TSV).  In either case, Hail will use only the `contig`, `start`, and `end` fields.  Each variant is evaluated against each line in the `.interval_list` file, and any match will mark the variant to be kept / excluded based on the presence of the `--keep` and `--remove` flags.  
 - _Note: "start" and "end" match positions inclusively, e.g. start <= position <= end_

2. `filtervariants` -- ".variant_list" file
 - Hail expects a .variant_list file to contain a variant per in line following format: `contig:pos:ref:alt1,alt2,...,altN`.  Variants in the dataset will be kept / excluded based on the presence of the `--keep` and `--remove` flags.

3. `filtersamples` -- ".sample_list" file
 - Hail expects a .sample_list file to contain a newline-delimited list of sample ids.  The `--keep` and `--remove` command-line flags will determine whether the list of samples is excluded or kept.  This file can contain sample IDs not present in the VDS.  

4. `filtergenotypes` -- no inclusion/exclusion files supported

## Using expressions

Hail provides powerful utility in filtering by allowing users to write their own boolean expressions on the command line, using the exposed genotype, sample, variant, and annotation objects.  This mode is used when the input to the `-c` command line argument does not match one of the expected inclusion/exclusion files extensions.

**Exposed namespaces:**
 - `filtersamples`: 
   - `s` (sample)
   - `sa` (sample annotation)
   - `gs` (genotype column [aggregable](HailExpressionLanguage.md#aggregables))
 - `filtervariants`:
   - `v` (variant)
   - `va` (variant annotation)
   - `gs` (genotype row [aggregable](HailExpressionLanguage.md#aggregables))
 - `filtergenotypes`:
   - `g` (genotype)
   - `s` (sample)
   - `sa` (sample annotation)
   - `v` (variant)
   - `va` (variant annotation)

A filtering expression is a computation involving the exposed objects that evaluates to a boolean (true / false).  These boolean expressions can be as simple or complicated as you like.  To learn more about these exposed objects, visit the documentation on [**Hail representation**](Representation.md) and [**the Hail expression language**](HailExpressionLanguage.md).
  
____
  
### Examples 

#### Using files

```
filtervariants -c 'file.interval_list' --keep
```

```
filtersamples -c 'file.sample_list' --remove
```

#### Using expressions

```
filtervariants -c 'v.contig == "5"' --keep
```

```
filtervariants -c 'va.pass' --keep
```

```
filtervariants -c '!va.pass' --remove
```

```
[after importvcf & splitmulti]
filtervariants -c 'va.info.AC[va.aIndex] > 1' --keep 
```

```
filtergenotypes -c 'g.ad(1).toDouble / g.dp < 0.2' --remove
```

```
filtersamples -c 'if ("DUMMY" ~ s.id) 
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
