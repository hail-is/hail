# Tutorial

For this tutorial, we will be analyzing data from the final phase of the [1000 Genomes Project](http://www.internationalgenome.org/about). 
The dataset is thoroughly described in [A global reference for human genetic variation. Nature 2015.](http://www.nature.com/nature/journal/v526/n7571/full/nature15393.html) 
The original data was down sampled to approximately 10,000 variants consisting of both rare and common variants. 
Sample information (population, super-population, sex) was obtained from the [1000 Genomes website](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_sample_info.xlsx).

### Prerequisites

(1) Read the [Overview page](overview.html) in the Documentation

(2) Install the following dependencies:

  - [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
  - [Java 1.8](https://www.google.com/search?q=download+java+8+jdk) (use `java -version` to see which version you currently have.  )
  - [Anaconda2 with Python 2.7](https://www.continuum.io/downloads)
  - [Spark 2.0.2](https://spark.apache.org/downloads.html)
  
If you are setting up a fresh Linux VM with Ubuntu installed (e.g. GCP or AWS), [this wiki](https://github.com/hail-is/hail/wiki/Install-Hail-dependencies-on-a-fresh-Ubuntu-VM) may help.

### Download and Install Hail    

Download and build Hail by entering the following commands (this will take a minute or two!): 

```
git clone https://github.com/hail-is/hail.git hail
cd hail/
./gradlew clean shadowJar
```

### Download Supplementary Data Files

Download the zip file (**Hail_Tutorial_Data-v1.tgz**) with [`wget`](https://www.google.com/search?q=install+wget) or [`curl`](https://www.google.com/search?q=install+curl):
 
```
wget https://storage.googleapis.com/hail-tutorial/Hail_Tutorial_Data-v1.tgz
```

Unzip the file with the following command:

```
tar -xvzf Hail_Tutorial_Data-v1.tgz --strip 1
```
      
The contents of the tar file are as follows:
  
  - 1000 Genomes Compressed VCF (down-sampled to 10K variants) -- **1000Genomes.ALL.coreExome10K-v1.vcf.bgz**
  - Sample Annotations -- **1000Genomes.ALL.coreExome10K-v1.sample_annotations**
  - LD-pruned SNP List -- **purcell5k.interval_list**


### Setting up the Spark / Hail ecosystem

The following commands need to be entered on the command line after installing Anaconda, downloading and unzipping the Spark installation, and installing java.

The below two commands may require modification based on where you cloned Hail and downloaded Spark.
```
export HAIL_HOME=~/hail  # or wherever you cloned Hail
export SPARK_HOME=~/spark-2.0.2-bin-hadoop2.7  # or wherever you unzipped Spark
```

These commands should require no modification.
```
export PYTHONPATH=$PYTHONPATH:$HAIL_HOME/python:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.3-src.zip
export SPARK_CLASSPATH=$HAIL_HOME/build/libs/hail-all-spark.jar
```

### Start an IPython interactive shell

Using the command `ipython` from the same directory where you extracted the tutorial files, start an IPython shell. You should see a window similar to the one shown below. If this doesn't work, Anaconda is not installed properly.

```text
hail@tutorial-vm:~$ ipython
Python 2.7.12 |Anaconda 4.2.0 (64-bit)| (default, Jul  2 2016, 17:42:40)
Type "copyright", "credits" or "license" for more information.

IPython 5.1.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]:
```

In this window, enter two commands:

    >>> from pyhail import *
    >>> hc = HailContext()

These two commands will run without error if everything is set up properly. If you see an error, make sure that the `export` variables are set correctly, that java is installed properly, and that Spark is the right version. If this step triggers a `Exception in thread "main" java.net.BindException`, check the [FAQ](https://hail.is/faq.html#how-do-i-fix-exception-in-thread-main-java.net.bindexception-cant-assign-requested-address-...) for a fix to this issue.


### Set Environment Variables

    >>> vcf = '1000Genomes.ALL.coreExome10K-v1.vcf.bgz'
    >>> sample_annotations = '1000Genomes.ALL.coreExome10K-v1.sample_annotations'
    >>> pruned_variants = 'purcell5k.interval_list'

## Import Data

Hail uses a fast and storage-efficient internal representation called a VDS (variant dataset). In order to use Hail for data analysis, data must first be imported to the VDS format. To do this, use the `import_vcf` method on `HailContext` to load the downsampled 1000 Genomes VCF (**1000Genomes.ALL.coreExome10K-v1.vcf.bgz**) into Hail. The VCF file is block-compressed (`.vcf.bgz`) which enables Hail to read the file in parallel. Reading a file that has not been block-compressed (`.vcf`, `.vcf.gz`) is _significantly_ slower and should be avoided. 

    >>> vds = hc.import_vcf(vcf)

Next, we use the `split_multi` method on `dataset` to split multi-allelic variants into biallelic variants. For example, the variant `1:1000:A:T,C` would become two variants: `1:1000:A:T` and `1:1000:A:C`.

    >>> vds = vds.split_multi()

Third, we use the `annotate_samples_table` method to load phenotypic information for each sample from the sample annotations file (**1000Genomes.ALL.coreExome10K-v1.sample_annotations**).

The first few lines of this file are reproduced below:
```
Sample  Population      SuperPopulation isFemale        PurpleHair      CaffeineConsumption
HG00096 GBR     EUR     False   False   77.0
HG00097 GBR     EUR     True    True    67.0
HG00098 GBR     EUR     False   False   83.0
HG00099 GBR     EUR     True    False   64.0
HG00100 GBR     EUR     True    False   59.0
HG00101 GBR     EUR     False   True    77.0
HG00102 GBR     EUR     True    True    67.0
```

The `root` flag tells Hail where to put the data read in from the file. When annotating samples, the first part of the root name should be `sa`. The second part can be anything you like. Here we have chosen `sa.pheno`. The `sample_expr` flag tells Hail how to select the sample ID. In this case, the column name containing the sample ID is `Sample`.

Finally, we need to pass configuration options. This is done using an object called `TextTableConfig`, which allows users to provide information about the data type of different columns, as well as set properties like header existence, comment character, or field delimiter. Here we need to read columns 'isFemale' and 'PurpleHair' as boolean values, and the 'CaffeineConsumption' column as a floating-point number. We can do this by passing an explicit type string to the `TextTableConfig` of the form 'isFemale: Boolean, PurpleHair: Boolean, CaffeineConsumption: Boolean', or we can pass the argument `impute=True` which enables Hail to guess the column types.

    >>> vds = vds.annotate_samples_table(sample_annotations, 
    >>>                                  root='sa.pheno', 
    >>>                                  sample_expr='Sample', 
    >>>                                  config=TextTableConfig(impute=True))

Lastly, we will `write` this dataset to disk so that we can read and process it more quickly (remember that the Hail representation is much faster than VCF!)  We'll then `read` this dataset, so that all future computations willbegin from the fast VDS rather than slow VCF.

    >>> out_path = '1kg.vds'
    >>> vds.write(out_path)
    >>> vds = hc.read(out_path)
    
Now we're ready to start exploring!  First, we'll print some simple statistics about the size of the dataset:

    >>> vds.count()
    
This method also takes an option `genotypes`, which can compute total call rate across all genotypes as well:

    >>> vds.count(genotypes=True)
    
<pre class="tutorial output" style="color: red">
Out[8]: {u'nGenotypes': 27786135L, u'nVariants': 10961L, u'nSamples': 2535, u'nCalled': 27417806L, u'callRate': 98.6744144156789}
</pre>

Note that the call rate is around 98.7% -- this will be revisited soon.  

We can also start by printing the types of all the annotations contained in the VDS. Many annotations came with the VCF, but we also added sample annotations above. Notice how the 6 sample annotation variables we loaded above are nested inside `sa.pheno` as defined by the `root` option in `annotate_samples table`.

    >>> vds.print_schema()

Next, we will list the populations that are present in our dataset and count the number of samples by phenotype using the Hail expression language and the `annotate_global_expr_by_sample` method.

The 1000 Genomes Super-Population codings are:

  - SAS = South Asian
  - AMR = Americas
  - EUR = European
  - AFR = African
  - EAS = East Asian

We'll first build up a list of annotation expressions, evaluate them on the dataset, and finally print the resulting global annotations.

    >>> expressions = [ 
    >>>   'global.populations = samples.map(s => sa.pheno.Population).collect().toSet',
    >>>   'global.superPopulations = samples.map(s => sa.pheno.SuperPopulation).collect().toSet',
    >>>   'global.nCases = samples.filter(s => sa.pheno.PurpleHair).count()',
    >>>   'global.nControls = samples.filter(s => !sa.pheno.PurpleHair).count()',
    >>>   'global.nSamples = samples.count()' ]
    >>> vds.annotate_global_expr_by_sample(expressions).show_globals()
    
<pre class="tutorial output" style="color: red">
Global annotations: `global' = {
  "populations" : [ "MSL", "GIH", "ASW", "JPT", "KHV", "CEU", "STU", "CDX", "BEB", "PUR", "ITU", "CLM", "GWD", "TSI", "ESN", "IBS", "PEL", "ACB", "YRI", "PJL", "CHS", "MXL", "CHB", "LWK", "FIN", "GBR" ],
  "superPopulations" : [ "SAS", "AMR", "EUR", "AFR", "EAS" ],
  "nCases" : 1300,
  "nControls" : 1235,
  "nSamples" : 2535
}
</pre>

It's also easy to count by population:

    >>> vds.annotate_global_expr_by_sample('global.count_by_pop = samples.map(s => sa.pheno.SuperPopulation).counter()').show_globals()

## QC

Before testing whether there is a genetic association for a given trait, the raw data must be filtered to remove genotypes that don't have strong evidence supporting the genotype call, samples that are outliers on key summary statistics across the dataset, and variants that have low mean genotype quality scores or don't follow a [Hardy Weinberg Equilibrium](https://en.wikipedia.org/wiki/Hardy–Weinberg_principle) distribution of genotype calls.


##### Filter Genotypes

We begin with an example of filtering genotypes based on allelic balance with the `filter_genotypes` method. Real data may require more complicated filtering expressions. To use this method, we must construct a **boolean expression** using the [Hail Expression Language](reference.html#HailExpressionLanguage). In this expression, `g` means the genotype, `v` is the variant, `s` is the sample, and annotations are accessible with `va`, `sa`, and `global`. 

We used the 'let ... in' functionality to define a new temporary variable `ab` for the allelic balance which is calculated from the allelic depth (`g.ad`) for each allele. Depending on the genotype call, we want the allelic balance to be between a given range. For example for heterozygote calls (`g.isHet`), we want the allelic balance to be between 0.25 and 0.75. Likewise, for a homozygote call (`g.isHomRef`), the allelic balance should be close to 0 indicating no evidence of the alternate allele. 

Additional methods on [Genotype](reference.html#genotype) are listed in the documentation.

    >>> filter_condition = '''let ab = g.ad[1] / g.ad.sum in
    >>>                     ((g.isHomRef && ab <= 0.1) || 
    >>>                     (g.isHet && ab >= 0.25 && ab <= 0.75) || 
    >>>                     (g.isHomVar && ab >= 0.9))'''
    >>> filtered_vds = vds.filter_genotypes(filter_condition)
    >>> filtered_vds.count(genotypes=True)

Notice that the call rate is just above 95%, whereas pre-filter it was 98.7%. Nearly 4% of genotypes failed this filter.

<pre class="tutorial output" style="color: red">
  nSamples             2,535
  nVariants           10,961
  nCalled         26,404,807
  callRate           95.029%
 </pre>
  
#### Filter Samples

Now that unreliable genotype calls have been filtered out, we can remove variants that have a low call rate before calculating summary statistics per sample with the `sample_qc` method. 
By removing poor-performing variants due to call rate, we can get a better picture of which samples are outliers on key quality control measures compared to what is expected.
The call rate was calculated by using Hail [aggregable](reference.html#aggregables) functionality to calculate the fraction of genotypes called per variant.
A description of all summary statistics that are calculated by the `sampleqc` command are available in the [documentation](commands.html#sampleqc).

    >>> filtered_vds_2 = (filtered_vds
    >>>     .filter_variants_expr('gs.fraction(g => g.isCalled) > 0.95')
    >>>     .sample_qc())

The sample QC method created more sample annotations. Let's print the schema again, and only the sample annotations this time:

    >>> filtered_vds_2.print_schema(sa=True)

It's very easy to export annotations to text files, as well. We will do that now:

    >>> filtered_vds_2.export_samples('sampleqc.txt', 'Sample = s.id, sa.qc.*')

There is a handy IPython magic command `%%sh` for a shell interpreter, which makes it very easy to peek at this file without leaving the IPython interpreter.

    >>> %%sh
    >>> head sampleqc.txt | cut -f 1,2,3,4,5,6,7,8,9,10

<pre class="tutorial output" style="color: red"> 
Sample	callRate	nCalled	nNotCalled	nHomRef	nHet	nHomVar	nSNP	nInsertion	nDeletion
HG02970	9.69313e-01	5433	172	3919	822	692	2206	0	0
NA19089	9.78947e-01	5487	118	4072	729	686	2101	0	0
NA18861	9.73417e-01	5456	149	3925	855	676	2207	0	0
HG02122	9.75022e-01	5465	140	4026	730	709	2148	0	0
NA20759	9.71097e-01	5443	162	4068	748	627	2002	0	0
HG00139	9.85370e-01	5523	82	4080	833	610	2053	0	0
NA12878	9.67351e-01	5422	183	4081	694	647	1988	0	0
HG02635	9.82337e-01	5506	99	3927	927	652	2231	0	0
NA19660	9.45049e-01	5297	308	3910	685	702	2089	0	0
</pre>

We can also analyze the results further using R. 
Below is an example of two variables that have been plotted (call rate and meanGQ). The red lines are cutoffs for filtering samples based on these two variables.

<img src="test.sampleqc.png">

We want to remove the samples that are outliers in the plots above, and we want to remove these from the VDS we had before variants were filtered -- it's possible that poor-quality samples decreased the call rate on variants we want to keep. There are many ways we can do this step, and we will demonstrate two. 

**Method 1:** export list of samples to keep from `filtered_vds_2`, and filter samples from `filtered_vds` based on this list.

    >>> (filtered_vds_2
    >>>     .filter_samples_expr('sa.qc.callRate >= 0.97 && sa.qc.gqMean >= 20')
    >>>     .export_samples('included_samples.txt', 's.id'))
    >>> filtered_vds_3 = filtered_vds.filter_samples_list('included_samples.txt')
    >>> print 'before filter: %d samples' % filtered_vds.num_samples()
    >>> print 'after filter: %d samples' % filtered_vds_3.num_samples()
    >>> method_1_kept_ids = filtered_vds_3.sample_ids()
    
**Method 2:** annotate the first VDS with the sample QC metrics we exported, and filter based on these metrics.

    >>> filtered_vds_3 = (filtered_vds
    >>>     .annotate_samples_table('sampleqc.txt', sample_expr='Sample', 
    >>>                             root='sa.qc', config=TextTableConfig(impute=True))
    >>>     .filter_samples_expr('sa.qc.callRate >= 0.97 && sa.qc.gqMean >= 20'))
    >>> print 'before filter: %d samples' % filtered_vds.num_samples()
    >>> print 'after filter: %d samples' % filtered_vds_3.num_samples()
    >>> method_2_kept_ids = filtered_vds_3.sample_ids()
    
Let's make sure these two methods give us the same samples:

    >>> method_1_kept_ids == method_2_kept_ids

Like we did when we first loaded the dataset, we can use the `annotate_global_expr_by_sample` method to count the number of samples by phenotype that remain in the dataset after filtering.

    >>> post_qc_exprs = [
    >>>     'global.postQC.nCases = samples.filter(s => sa.pheno.PurpleHair).count()',
    >>>     'global.postQC.nControls = samples.filter(s => !sa.pheno.PurpleHair).count()' ]
    >>> filtered_vds_3.annotate_global_expr_by_sample(post_qc_exprs).show_globals()     

<pre class="tutorial output" style="color: red">
Global annotations: `global' = {
  "postQC" : {
    "nCases" : 840,
    "nControls" : 806,
  }
}
</pre>

#### Filter Variants

We now have `filtered_vds_3`, a VDS where both poor-performing genotypes and samples have been removed. We can start exploring variant metrics using the `variant_qc` method, which computes generates QC summary statistics as annotations per variant. We will also export these annotations to a text file.

We use the string `va.qc.*` to specify that all annotations in the struct `va.qc` should be included as columns. We could also have written the export expression above as `Variant = v, va.qc.*` in which case the Variant column would have the form "Chrom:Pos:Ref:Alt".

    >>> filtered_vds_3 = filtered_vds_3.variant_qc()
    >>> filtered_vds_3.print_schema(va=True)
    >>> filtered_vds_3.export_variants('variantqc.tsv',
    >>>                                'Chrom=v.contig, Pos=v.start, Ref=v.ref, Alt=v.alt, va.qc.*')

We've used R to make histograms of 4 summary statistics (call rate, minor allele frequency, mean GQ, and [Hardy Weinberg Equilibrium P-value](https://en.wikipedia.org/wiki/Hardy–Weinberg_principle)). Notice how the histogram for HWE does not look as one would expect (most variants should have a p-value close to 1). This is because there are 5 populations represented in this dataset and the p-value we calculated includes all populations.
<img src="test.variantqc.png">

To compute the HWE p-value by population, we use the `annotate_variants expr` method to programmatically compute Hardy Weinberg Equilibrium for each population.
For each variant, we filter the genotypes to only those genotypes from the population of interest using a filter function on the [genotype aggregable](reference.html#aggregables) and then calculate the Hardy-Weinberg Equilibrium p-value using the [`hardyWeinberg`](reference.html#aggreg_hwe) function on the filtered genotype aggregable. 

The `persist` method we've added caches the dataset in its current state on memory/disk, so that downstream processing will be faster.

The results of `print_schema` show we have added new fields in the variant annotations for HWE p-values for each population.

    >>> hwe_expressions = [
    >>>     'va.hweByPop.hweEUR = gs.filter(g => sa.pheno.SuperPopulation == "EUR").hardyWeinberg()',
    >>>     'va.hweByPop.hweSAS = gs.filter(g => sa.pheno.SuperPopulation == "SAS").hardyWeinberg()',
    >>>     'va.hweByPop.hweAMR = gs.filter(g => sa.pheno.SuperPopulation == "AMR").hardyWeinberg()',
    >>>     'va.hweByPop.hweAFR = gs.filter(g => sa.pheno.SuperPopulation == "AFR").hardyWeinberg()',
    >>>     'va.hweByPop.hweEAS = gs.filter(g => sa.pheno.SuperPopulation == "EAS").hardyWeinberg()' ]
    >>> filtered_vds_3 = filtered_vds_3.annotate_variants_expr(hwe_expressions)
    >>> filtered_vds_3.persist()
    >>> filtered_vds_3.print_schema(va=True)

We've got quite a few variant annotations now! Notice that the results of these annotation statements are structs containing two elements:

<pre class="tutorial output" style="color: red">
Variant annotation schema:
...
        hweEUR: Struct {
            rExpectedHetFrequency: Double,
            pHWE: Double
        },
        hweSAS: Struct {
            rExpectedHetFrequency: Double,
            pHWE: Double
        },
        hweAMR: Struct {
            rExpectedHetFrequency: Double,
            pHWE: Double
        },
        hweAFR: Struct {
            rExpectedHetFrequency: Double,
            pHWE: Double
        },
        hweEAS: Struct {
            rExpectedHetFrequency: Double,
            pHWE: Double
        }
    }
}
</pre>
 
Now that the variant annotations contain a population-specific p-value for HWE, we can filter variants based on passing HWE in each population. The results of the `count` method confirms that by calculating HWE p-values in each population separately, we only filter out 826 variants compared to 7098 variants before.

The `persist` method we've added before `count` caches the dataset in its current state on memory/disk, so that downstream processing will be faster.

    >>> hwe_filter_expression = '''
    >>>     va.hweByPop.hweEUR.pHWE > 1e-6 && 
    >>>     va.hweByPop.hweSAS.pHWE > 1e-6 && 
    >>>     va.hweByPop.hweAMR.pHWE > 1e-6 && 
    >>>     va.hweByPop.hweAFR.pHWE > 1e-6 && 
    >>>     va.hweByPop.hweEAS.pHWE > 1e-6 '''
    >>> hwe_filtered_vds = filtered_vds_3.filter_variants_expr(hwe_filter_expression)
    >>> hwe_filtered_vds.count()


<pre class="tutorial output" style="color: red">
Out[49]: {u'nSamples': 1646, u'nVariants': 10135L, u'nGenotypes': 16682210L}
</pre>

Lastly we use the `filter_variants expr`method to keep variants with a mean GQ greater than or equal to 20.

    >>> final_filtered_vds = hwe_filtered_vds.filter_variants_expr('va.qc.gqMean >= 20')
    >>> final_filtered_vds.count()

<pre class="tutorial output" style="color: red">
{u'nSamples': 1646, u'nVariants': 9949L, u'nGenotypes': 16376054L}
</pre>
  
We can see we have filtered out 1,012 total variants from the dataset.

#### Sex Check

One of the most important QC metrics is to ensure that the reported sex of the samples is consistent with sex chromosome ploidy estimated from the genetic data. 
A high sex check failure rate can indicate that sample swaps may have occurred.

First, we count how many X chromosome variants are in the original dataset, and find there are 273.

    >>> vds.filter_variants_expr('v.contig == "X"').num_variants()
  
However, after variant QC, the number of X chromosome variants went from 273 to 10 (not enough for a sex check!)

    >>> final_filtered_vds.filter_variants_expr('v.contig == "X"').num_variants()

This happened because the HWE p-values for the X chromosome should not include male samples in the calculation as they only have two possible genotypes (HomRef or HomVar). We're going to have to go back to the pre-hwe-filtered `filtered_vds_3` and modify how we calculate HWE. We use a conditional expression so that when a variant is on the X chromosome, we only include female samples in the calculation. We can also use the same `hwe_filter_expression` from above.

    >>> sex_aware_hwe_exprs = [ 
    >>>      '''va.hweByPop.hweEUR = 
    >>>         if (v.contig != "X") 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "EUR").hardyWeinberg() 
    >>>         else 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "EUR" && sa.pheno.isFemale).hardyWeinberg()''',
    >>>      '''va.hweByPop.hweSAS = 
    >>>         if (v.contig != "X") 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "SAS").hardyWeinberg() 
    >>>         else 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "SAS" && sa.pheno.isFemale).hardyWeinberg()''',
    >>>      '''va.hweByPop.hweAMR = 
    >>>         if (v.contig != "X") 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "AMR").hardyWeinberg() 
    >>>         else 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "AMR" && sa.pheno.isFemale).hardyWeinberg()''',
    >>>      '''va.hweByPop.hweAFR = 
    >>>         if (v.contig != "X") 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "AFR").hardyWeinberg() 
    >>>         else 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "AFR" && sa.pheno.isFemale).hardyWeinberg()''',
    >>>      '''va.hweByPop.hweEAS = 
    >>>         if (v.contig != "X") 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "EAS").hardyWeinberg() 
    >>>         else 
    >>>           gs.filter(g => sa.pheno.SuperPopulation == "EAS" && sa.pheno.isFemale).hardyWeinberg()''' ]
    >>> hwe_filtered_vds_fixed = (filtered_vds_3
    >>>     .annotate_variants_expr(sex_aware_hwe_exprs)
    >>>     .filter_variants_expr(hwe_filter_expression)
    >>>     .persist())
    >>>
    >>> print 'total variants = %s' % hwe_filtered_vds_fixed.num_variants()
    >>> print 'X chromosome variants = %s' % hwe_filtered_vds_fixed.filter_variants_expr('v.contig == "X"').num_variants()
  
To do a sex check, first we use the `impute_sex` method with a minimum minor allele frequency threshold `maf_threshold` argument of 0.05 to determine the genetic sex of a sample based on the inbreeding coefficient.
`impute_sex` adds new sample annotations for whether a sample is predicted to be female to `sa.imputesex.isFemale`. 
We can then create a new sample annotation (`sa.sexcheck`) which compares whether the imputed sex (`sa.imputesex.isFemale`) is the same as the reported sex (`sa.pheno.isFemale`).

    >>> sex_check_vds = (hwe_filtered_vds_fixed
    >>>     .impute_sex(maf_threshold=0.05)
    >>>     .annotate_samples_expr('sa.sexcheck = sa.pheno.isFemale == sa.imputesex.isFemale'))
    >>> total_samples = sex_check_vds.num_samples()
    >>> sex_check_passes = sex_check_vds.filter_samples_expr('sa.sexcheck').num_samples()
    >>> print 'total samples: %s' % total_samples
    >>> print 'sex_check_passes: %s' % sex_check_passes
    
We removed 567 samples where the genetic sex does not match the reported sex. This is an extremely high sex check failure rate! To figure out why this happened, we can use a Hail expression to look at the values that `sa.sexcheck` takes.

    >>> sex_check_vds.annotate_global_expr_by_sample(
    >>>     'global.sexcheckCounter = samples.map(s => sa.sexcheck).counter()').show_globals()

Aha! While we only have 3 'false' sexcheck values (which means that the samples were imputed as male but reported as female, or the reverse), we have 564 missing sexcheck values. Since `pheno.isFemale` is never missing (see the sample annotations file), this means that there were 564 samples that could not be imputed as male or female. This is probably because our vcf has so few variants -- ~200 variants is not sufficient to impute reliably. Instead of filtering out samples whose sexcheck is missing, we should probably keep them. We can write a filter expression that does this:
 
    >>> sex_check_filtered_vds = sex_check_vds.filter_samples_expr('sa.sexcheck || isMissing(sa.sexcheck)').persist()
    >>> print 'samples after filter: %s' % sex_check_filtered_vds.num_samples()

## PCA

To account for population stratification in association testing, we use principal component analysis to compute covariates that are proxies for genetic similarity.
For PCA to work, we need an independent set of SNPs. The text file **purcell5k.interval_list** contains a list of independent variants.
To calculate principal components, we first use the `filter_variants_intervals` method to only keep SNPs from the **purcell5k.interval_list**. Next, we use the `pca` method to calculate the first 10 principal components. The results are stored in sample annotations rooted by the `scores` parameter. Lastly, we export the sample annotations to a text file so that we can plot the principal components and color the points by their population group.

    >>> pca_vds = (sex_check_filtered_vds.filter_variants_intervals('purcell5k.interval_list')
    >>>     .pca(scores='sa.pca'))
    >>> pca_vds.export_samples('pcaPlusPopulation.tsv', 
    >>>     'Sample=s, SuperPopulation=sa.pheno.SuperPopulation,'
    >>>     'Population=sa.pheno.Population, sa.pca.*')

Here are some examples plotted using R:
<img src="test.pcaPlot.png">

## Association Testing

Now that we have a QC'd dataset with principal components calculated and phenotype information added, we can test for an association between the genetic variants and the phenotypes of PurpleHair (dichotomous) and CaffeineConsumption (continuous).

#### Linear Regression with Covariates

We are going to run linear regression on the `sex_check_filtered_vds` VDS. First, we will filter out variants with a minor allele frequency less than 5% (also include 95% as it's possible for the minor allele to be the reference allele). Next, we use the `linreg` method, specifying the response variable `y` to be the sample annotation for CaffeineConsumption `sa.pheno.CaffeineConsumption`. We also define 4 covariates to correct for: `sa.pca.PC1`, `sa.pca.PC2`, `sa.pca.PC3`, `sa.pheno.isFemale`. The results of linear regression are stored as variant annotations and can be accessed with the root name `va.linreg`. Lastly we export these results to a text file for making a Q-Q plot in R.

    >>> analysis_ready_vds = (sex_check_filtered_vds
    >>>     .filter_variants_expr('va.qc.AF > 0.05 && va.qc.AF < 0.95')
    >>>     .annotate_samples_vds(pca_vds, code='sa.pca = vds.pca'))
    >>> (analysis_ready_vds
    >>>     .linreg('sa.pheno.CaffeineConsumption', 
    >>>             covariates='sa.pca.PC1, sa.pca.PC2, sa.pca.PC3, sa.pheno.isFemale')
    >>>     .export_variants('linreg.tsv', 'Variant=v, va.linreg.*'))
    
<img src="test.linreg.qq.png">

#### Logistic Regression with Covariates

We can start from our `analysis_read_vds` once again. The logistic regression method is similar, but it also takes a test type argument. We will use the Wald test.

    >>> (analysis_ready_vds
    >>>     .logreg(test='wald', y='sa.pheno.PurpleHair',
    >>>             covariates='sa.pca.PC1, sa.pca.PC2, sa.pca.PC3, sa.pheno.isFemale')
    >>>     .export_variants('logreg.tsv', 'Variant=v, PVAL = va.logreg.wald.pval'))

<img src="test.logreg.qq.png">

#### Fisher's Exact Test for Rare Variants

We'll start with `sex_check_filtered_vds` here (our `analysis_ready_vds` isn't so analysis-ready for rare variant tests because we filtered them all out!). This time we will filter out common variants with a minor allele frequency greater than 5% and less than 95%, so we're left with rare variants. Next we will annotate variants with 4 metrics about the aggregate statistics of the samples at each position. These new variant annotations can be used as inputs to the `fet`, or Fisher Exact Test, function which takes 4 integers representing a 2x2 contingency table. We define the output of the [`fet`](reference.html#fet) function will go into the variant annotations keyed by `va.fet`.
Lastly, we export the results to a text file and make a Q-Q plot in R. 

    >>> rare_variant_annotations = [
    >>>     '''va.minorCase = 
    >>>         gs.filter(g => sa.pheno.PurpleHair && g.isHet).count() +
    >>>         2 * gs.filter(g => sa.pheno.PurpleHair && g.isHomVar).count()''',
    >>>     '''va.minorControl = 
    >>>         gs.filter(g => !sa.pheno.PurpleHair && g.isHet).count() + 
    >>>         2 * gs.filter(g => !sa.pheno.PurpleHair && g.isHomVar).count()''',
    >>>     '''va.majorCase = 
    >>>         gs.filter(g => sa.pheno.PurpleHair && g.isHet).count() +
    >>>         2 * gs.filter(g => sa.pheno.PurpleHair && g.isHomRef).count()''',
    >>>     '''va.majorControl = 
    >>>         gs.filter(g => !sa.pheno.PurpleHair && g.isHet).count() +
    >>>         2 * gs.filter(g => !sa.pheno.PurpleHair && g.isHomRef).count()''' ]
    >>>
    >>> (sex_check_filtered_vds
    >>>     .filter_variants_expr('va.qc.AF <= 0.05 || va.qc.AF >= 0.95')
    >>>     .annotate_variants_expr(rare_variant_annotations)
    >>>     .annotate_variants_expr('''va.fet = 
    >>>                                 fet(va.minorCase.toInt, va.minorControl.toInt,
    >>>                                     va.majorCase.toInt, va.majorControl.toInt)''')
    >>>     .export_variants('fisherExactTest.tsv', 'Variant = v, va.fet.*'))

<img src="test.fet.qq.png">
