## <a class="jumptarget" name="annotations"></a> Annotations 

#### How can I find the current schema of variant, sample, and global annotations?

Use the [`printschema`](commands.html#printschema) command. By default the schemas for variant, sample, and global annotations are printed to stdout.

To specify which schemas you would like to view use the following flags: variant annotations (`--va`), sample annotations (`--sa`), global annotations (`--global`).

Schemas can also be output to a file using the `-o` flag.

```
printschema
```

#### How do I access an annotation name with white-space in the Hail Expression Language?

Put the annotation name in back ticks.

```
annotateglobal expr -c 'global.`my variable` = global.`lof count`'
```

#### How do I count the number of samples matching a phenotype annotation?

```
annotateglobal expr -c '
    global.nMales = samples.count(sa.pheno.sex == "Male"),
    global.nFemales = samples.count(sa.pheno.sex == "Female"),
    global.nSamples = samples.count(true)'
```


#### How do I import annotations from a PLINK fam file?

Use the [`annotatesamples fam`](commands.html#annotatesamples_fam) command. For quantitative phenotypes, make sure to add the `-q` flag!

```
annotatesamples fam -i myStudy.fam -q
```


#### How do I make a new annotation that combines annotations currently in the dataset?

```
annotatevariants expr -c 'va.combAnnot = va.annot1 + ":" + va.annot2'
```


#### How do I create an annotation for only a subset of samples based on an existing annotation?

```
annotatesamples expr -c 'if (sa.pheno.cohortName == "cohort1") sa.pheno.bloodPressure else NA: Double'
```

For optimal performance, make sure the Type defined after `NA:` is the same as the type of the response variable `sa.pheno.bloodPressure`.