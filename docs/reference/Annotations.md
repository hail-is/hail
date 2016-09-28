# Annotations in Hail

Annotations are a core aspect of data representation in Hail.  At their simplest, they are data associated with either samples, variants, or global for the dataset.  In practice, they can be thought of as SQL tables indexed by the same.  Many of Hail's modules either produce or query annotations.  

Annotations are accessed using the key prefixes for sample, variants, and global, which are `sa`, `va`, `global` (**s**ample **a**nnotations and **v**ariant **a**nnotations).  Identifiers follow and layers of the annotation structure are delimited by periods.  Annotations nest arbitrarily, and while most come in the form of `va.pass` or `sa.qc.callRate`, some are more complicated like `va.vep.colocated_variants[<index>].aa_maf`.

If at any time you want to know what annotations are stored within Hail, use the [`printschema`](commands.html#printschema) module.  This module will print the schema for both samples and variants in a readable form.

Additionally, users can import annotations from a variety of files, or define new annotations as functions of the exposed data structures. 

## Commands that Generate Sample Annotations

#### User-Defined:

 - [`annotatesamples expr`](commands.html#annotatesamples_expr) -- annotate samples programatically using the [Hail Expression Language](#HailExpressionLanguage)
 - [`annotatesamples fam`](commands.html#annotatesamples_fam) -- annotate samples from a [PLINK .fam file](https://www.cog-genomics.org/plink2/formats#fam)
 - [`annotatesamples list`](commands.html#annotatesamples_list) -- annotate samples with whether they are present in a list of samples
 - [`annotatesamples table`](commands.html#annotatesamples_table) -- annotate samples with information in tabular format


#### Hail-Generated:

 - [`imputesex`](commands.html#imputesex) -- imputed sex of samples
 - [`sampleqc`](commands.html#sampleqc) -- sample qc summary statistics


## Commands that Generate Variant Annotations

#### User-Defined:

 - [`annotatevariants bed`](commands.html#annotatevariants_bed) -- annotate variants with information from a [UCSC BED file](https://genome.ucsc.edu/FAQ/FAQformat.html#format1))
 - [`annotatevariants expr`](commands.html#annotatevariants_expr) -- annotate variants programatically using the [Hail Expression Language](#HailExpressionLanguage)
 - [`annotatevariants intervals`](commands.html#annotatevariants_intervals) -- annotate variants with whether they are present in a list of intervals
 - [`annotatevariants loci`](commands.html#annotatevariants_loci) -- annotate variants with data in tabular format by chromosome & position
 - [`annotatevariants table`](commands.html#annotatevariants_table) -- annotate variants with data in tabular format by chromosome, position, ref, & alt
 - [`annotatevariants vcf`](commands.html#annotatevariants_vcf) -- annotate variants with the INFO field from a secondary VCF file
 - [`annotatevariants vds`](commands.html#annotatevariants_vds) -- annotate variants with variant annotations (`va`) from a second VDS file
 
 - [`importannotations_table`](commands.html#importannotations_table)
     - annotate variants with data in tabular format
     - results in sites-only VDS
     - can directly impute types


#### Hail-Generated:

 - [`importbgen`](commands.html#importbgen) -- rsID and variant ID
 - [`importgen`](commands.html#importgen) -- rsID and variant ID
 - [`importvcf`](commands.html#importvcf) -- INFO field from VCF
 - [`linreg`](commands.html#linreg) -- linear regression test statistics
 - [`logreg`](commands.html#logreg) -- logistic regression test statistics
 - [`splitmulti`](commands.html#splitmulti) -- information on whether a variant was originally multi-allelic
 - [`variantqc`](commands.html#variantqc) -- variant qc summary statistics
 - [`vep`](commands.html#vep) -- Variant Effect Predictor output


## Commands that Generate Global Annotations

#### User-Defined:

 - [`annotateglobal expr`](commands.html#annotateglobal_expr) -- Generate global annotations using the Hail expr language, including the ability to aggregate sample and variant statistics
 - [`annotateglobal list`](commands.html#annotateglobal_list) -- Read a file to global annotations as an `Array[String]` or `Set[String]`.
 - [`annotateglobal table`](commands.html#annotateglobal_table) -- Read a file to global annotations as an `Array[Struct]` using Hail's table parsing module.