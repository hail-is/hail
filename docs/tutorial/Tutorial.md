# Tutorial

For this tutorial, we will be analyzing data from the final phase of the [1000 Genomes Project](http://www.internationalgenome.org/about). 
The dataset is thoroughly described in [A global reference for human genetic variation. Nature 2015.](http://www.nature.com/nature/journal/v526/n7571/full/nature15393.html) 
The original data was down sampled to approximately 10,000 variants consisting of both rare and common variants. 
Sample information (population, super-population, sex) was obtained from the [1000 Genomes website](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_sample_info.xlsx).

## Setup

This tutorial can be done in two different ways: 

 - [Jupyter/iPython Notebook](http://jupyter.readthedocs.io/en/latest/)
 - Manually copying commands into a terminal window
 

### Prerequisites

(1) Read the [Overview page](overview.html) in the Documentation

(2) Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Java 1.8](https://www.google.com/search?q=download+java+8+jdk)


### Optional -- Install Jupyter/iPython Notebook
 
(1) Install [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html).         
             
(2) Install the [Bash Kernel for Jupyter](https://github.com/takluyver/bash_kernel).

(3) Download a tarball containing the Jupyter/iPython Notebook file using [wget](https://www.google.com/search?q=install+wget) or [curl](https://www.google.com/search?q=install+curl).

```
wget https://hail.is/Hail_Tutorial-v1.tgz
```

(4) Decompress the tarball using [tar](https://www.google.com/search?q=decompress+tarball)

```
tar -xvzf Hail_Tutorial-v1.tgz
```

(5) Open Jupyter from the command line which will open a new browser window. You should see a notebook loaded with the tutorial.     
    
```
jupyter notebook --no-mathjax Hail_Tutorial-v1.ipynb
```

### Download and Install Hail    

Download and build Hail by entering the following commands (this will take a minute or two!): 

```
git clone https://github.com/hail-is/hail.git hail-tutorial
cd hail-tutorial/
./gradlew clean installDist
alias hail="`pwd`/build/install/hail/bin/hail"
```

If Hail was built successfully, you should be able to see the message `hail: fatal: no commands given` when entering `hail` on the command line:
    
```
hail
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

### Set Environment Variables

```
vcf=1000Genomes.ALL.coreExome10K-v1.vcf.bgz
sampleAnnotations=1000Genomes.ALL.coreExome10K-v1.sample_annotations
prunedVariants=purcell5k.interval_list
```

## Import Data

Hail uses a fast and storage-efficient internal representation called a VDS (variant dataset). 
In order to use Hail for data analysis, data must first be imported to the VDS format. 
To do this, use the [`importvcf`](commands.html#importvcf) command to load the downsampled 1000 Genomes VCF (**1000Genomes.ALL.coreExome10K-v1.vcf.bgz**) into Hail. 
The VCF file is block-compressed (`.vcf.bgz`) which enables Hail to read the file in parallel. 
Reading a file that has not been block-compressed (`.vcf`, `.vcf.gz`) is significantly slower and should be avoided. 

Next, we use the [`splitmulti`](commands.html#splitmulti) command to split multi-allelic variants into separate variants. 
For example, the variant `1:1000:A:T,C` would become two variants: `1:1000:A:T` and `1:1000:A:C`.

Third, we use the [`annotatesamples table`](commands.html#annotatesamples_table) command to load phenotypic information for each sample from the sample annotations file (**1000Genomes.ALL.coreExome10K-v1.sample_annotations**).
The `--root` flag tells Hail where to put the data read in from the file. 
When annotating samples, the first part of the root name should be `sa`. 
The second part can be anything you like. Here we have chosen `sa.pheno`. 
The `-e` flag tells Hail which column contains the sample ID. 
In this case, the column name containing the sample ID is `Sample`. 
Lastly, the `--types` flag tells Hail what type each of the columns are where the format is `columnName: Type`. 
Hail can also infer the input types of each column by replacing the `-t` option with `--impute`. 
The `isFemale` column is coded such that females are `True` and males are `False`. 
Two phenotypes were randomly generated: `PurpleHair` is a dichotomous variable (`Type = Boolean`) and `CaffeineConsumption` is a continuous variable (`Type = Double`).

Lastly, we [`write`](commands.html#write) the imported data to Hail's VDS format (**test.raw.vds**) and perform a [`count`](commands.html#count) operation to print out summary statistics about the dataset.

```
hail importvcf $vcf \
    \
    splitmulti \
    \
    annotatesamples table --root sa.pheno -e Sample \
        --types 'Population: String, SuperPopulation: String, isFemale: Boolean, 
                 PurpleHair: Boolean, CaffeineConsumption: Double' \
        --input $sampleAnnotations \
    \
    write -o test.raw.vds \
    \
    count -g
```

<pre class="tutorial output" style="color: red">
  nSamples             2,535
  nVariants           10,961
  nCalled         27,417,806
  callRate           98.674%
</pre>

If this step triggers a `Exception in thread "main" java.net.BindException`, check the [FAQ](https://hail.is/faq.html#how-do-i-fix-exception-in-thread-main-java.net.bindexception-cant-assign-requested-address-...) for a fix to this issue.

We can print the schema of the sample annotations that were loaded above with the [`printschema`](commands.html#printschema) command and the `--sa` flag. 
Notice how the 6 sample annotation variables we loaded above are nested inside `sa.pheno` as defined by the `--root` flag in the [`annotatesamples table`](commands.html#annotatesamples_table) command.

```
hail read -i test.raw.vds \
    \
    printschema --sa
```

<pre class="tutorial output" style="color: red">
Sample annotation schema:
sa: Struct {
    pheno: Struct {
        Sample: String,
        Population: String,
        SuperPopulation: String,
        isFemale: Boolean,
        PurpleHair: Boolean,
        CaffeineConsumption: Double
    }
}
</pre>

Lastly, we can see which populations are present in our dataset and count the number of samples in the dataset by phenotype using the [`annotateglobal expr`](commands.html#annotateglobal_expr) and [`showglobals`](commands.html#showglobals) commands. 
The 1000 Genomes Super-Population codings are:

  - SAS = South Asian
  - AMR = Americas
  - EUR = European
  - AFR = African
  - EAS = East Asian

```
hail read -i test.raw.vds \
    \
    annotateglobal expr -c 'global.populations = samples.map(s => sa.pheno.Population).collect().toSet,
                            global.superPopulations = samples.map(s => sa.pheno.SuperPopulation).collect().toSet,
                            global.nCases = samples.filter(s => sa.pheno.PurpleHair).count(),
                            global.nControls = samples.filter(s => !sa.pheno.PurpleHair).count(),
                            global.nSamples = samples.count()' \
    \
    showglobals
```

<pre class="tutorial output" style="color: red">
Global annotations: `global' = {
  "populations" : [ "MSL", "GIH", "ASW", "JPT", "KHV", "CEU", "STU", "CDX", "BEB", "PUR", "ITU", "CLM", "GWD", "TSI", "ESN", "IBS", "PEL", "ACB", "YRI", "PJL", "CHS", "MXL", "CHB", "LWK", "FIN", "GBR" ],
  "superPopulations" : [ "SAS", "AMR", "EUR", "AFR", "EAS" ],
  "nCases" : 1300,
  "nControls" : 1235,
  "nSamples" : 2535
}
</pre>

## QC

Before testing whether there is a genetic association for a given trait, the raw data must be filtered to remove genotypes that don't have strong evidence supporting the genotype call, samples that are outliers on key summary statistics across the dataset, and variants that have low mean genotype quality scores or don't follow a [Hardy Weinberg Equilibrium](https://en.wikipedia.org/wiki/Hardy–Weinberg_principle) distribution of genotype calls.


##### Filter Genotypes

Here is an example of filtering genotypes based on allelic balance with the [`filtergenotypes`](commands.html#filtergenotypes) command. Real data may require more complicated filtering expressions. 
First, we tell Hail which VDS file to [`read`](commands.html#read) from. 
Next, we construct a **boolean expression** using the [Hail Expression Language](reference.html#HailExpressionLanguage) that is designated by the `-c` flag. 
Since we have specified the `--keep` flag, genotypes with an expression evaluating to True will be kept while genotypes evaluating to False will be removed. 
We used the 'let ... in' functionality to define a new temporary variable `ab` for the allelic balance which is calculated from the allelic depth (`g.ad`) for each allele. 
Depending on the genotype call, we want the allelic balance to be between a given range. 
For example for heterozygote calls (`g.isHet`), we want the allelic balance to be between 0.25 and 0.75. 
Likewise, for a homozygote call (`g.isHomRef`), the allelic balance should be close to 0 indicating no evidence of the alternate allele.
The resulting comparisons of allele balance per genotype class result in a boolean expression which is then used by the [`filtergenotypes`](commands.html#filtergenotypes) command.
Additional methods for [Genotypes](reference.html#genotype) are listed in the documentation.

```
hail read -i test.raw.vds \
    filtergenotypes --keep -c 'let ab = g.ad[1] / g.ad.sum 
                               in ((g.isHomRef && ab <= 0.1) || 
                                   (g.isHet && ab >= 0.25 && ab <= 0.75) || 
                                   (g.isHomVar && ab >= 0.9))' \
    write -o test.filtergeno.vds \
    \
    count -g
```

<pre class="tutorial output" style="color: red">
  nSamples             2,535
  nVariants           10,961
  nCalled         26,404,807
  callRate           95.029%
 </pre>
  
Comparing the results of the [`count`](commands.html#count) command before and after filtering genotypes, we filtered out 1,012,999 genotypes that did not meet the allelic balance criteria we specified.

#### Filter Samples

Now that unreliable genotype calls have been filtered out, we can remove variants that have a low call rate before calculating summary statistics per sample with the [`sampleqc`](commands.html#sampleqc) command. 
By removing poor-performing variants due to call rate, we can get a better picture of which samples are outliers on key quality control measures compared to what is expected.
The call rate was calculated by defining a temporary variable `callRate` with the let...in syntax and then using [aggregable](reference.html#aggregables) functionality to calculate the call rate.
A description of all summary statistics that are calculated by the `sampleqc` command are available in the [documentation](commands.html#sampleqc).

```
hail read -i test.filtergeno.vds \
    \
    filtervariants expr --keep -c 'let callRate = gs.filter(g => g.isCalled).count() / gs.count() 
                                   in callRate >= 0.95' \
    \
    sampleqc -o test.sampleqc.tsv
```

After calculating the sample qc summary statistics, we can look at the output text file **test.sampleqc.tsv**

```
head test.sampleqc.tsv | cut -f 1,2,3,4,5,6,7,8,9,10
```

<pre class="tutorial output" style="color: red"> 
Sample	callRate	nCalled	nNotCalled	nHomRef	nHet	nHomVar	nSNP	nInsertion	nDeletion
HG02970	9.6931e-01	5433	172	3919	822	692	1514	0	0
NA19089	9.7895e-01	5487	118	4072	729	686	1415	0	0
NA18861	9.7342e-01	5456	149	3925	855	676	1531	0	0
HG02122	9.7502e-01	5465	140	4026	730	709	1439	0	0
NA20759	9.7110e-01	5443	162	4068	748	627	1375	0	0
HG00139	9.8537e-01	5523	82	4080	833	610	1443	0	0
NA12878	9.6735e-01	5422	183	4081	694	647	1341	0	0
HG02635	9.8234e-01	5506	99	3927	927	652	1579	0	0
NA19660	9.4505e-01	5297	308	3910	685	702	1387	0	0
</pre>

We can also analyze the results further using R. 
Below is an example of two variables that have been plotted (call rate and  meanGQ). The red lines are cutoffs for filtering samples based on these two variables.

<img src="test.sampleqc.png">

To remove the samples that are outliers in the plots above, we first start by reading the VDS file where only the genotypes have been filtered (test.filtergeno.vds). 
Then we use the [`annotatesamples table`](commands.html#annotatesamples_table) command to add the summary statistics calculated from the [`sampleqc`](commands.html#sampleqc) command above to the root path `sa.qc`.
Once the sample qc annotations have been loaded into the VDS, we can use the [`filtersamples expr`](commands.html#filtersamples_expr) command to keep the samples that meet the filtering criteria defined in the plots above.
Lastly, we write out the filtered dataset to a new VDS file (**test.filtersamples.vds**). 
Notice that we didn't add the `-g` flag to [`count`](commands.html#count) this time because counting genotypes takes more time and we weren't filtering genotypes in this set of commands.


```
hail read -i test.filtergeno.vds \
    \
    annotatesamples table -e Sample -r sa.qc --impute -i test.sampleqc.tsv \
    \
    filtersamples expr --keep -c 'sa.qc.callRate >= 0.97 && sa.qc.gqMean >= 20' \
    \
    write -o test.filtersamples.vds \
    \
    count
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants           10,961
</pre>
  
Like we did when we first loaded the dataset, we can use the [`annotateglobal expr`](commands.html#annotateglobal_expr) and [`showglobals`](commands.html#showglobals) commands to count the number of samples by phenotype that remain in the dataset after filtering.


```
hail read -i test.filtersamples.vds \
    \
    annotateglobal expr -c 'global.postQC.nCases = 
                                samples.filter(s => sa.pheno.PurpleHair).count(), 
                            global.postQC.nControls = 
                                samples.filter(s => !sa.pheno.PurpleHair).count(), 
                            global.postQC.nSamples = 
                                samples.count()' \
    \
    showglobals
```

<pre class="tutorial output" style="color: red">
Global annotations: `global' = {
  "postQC" : {
    "nCases" : 840,
    "nControls" : 806,
    "nSamples" : 1646
  }
}
</pre>

We have filtered out 889 samples from the original dataset.

#### Filter Variants

Starting from a VDS where both poor-performing genotypes and samples have been removed (**test.filtersamples.vds**), we can use the [`variantqc`](commands.html#variantqc) command to compute numerous QC summary statistics per variant and output the results to a text file called **test.variantqc.tsv**. 
The fields in this file are described in the documentation for [`variantqc`](commands.html#variantqc).

```
hail read -i test.filtersamples.vds \
    \
    variantqc -o test.variantqc.tsv
```

We've used R to make histograms of 4 summary statistics (call rate, minor allele frequency, mean GQ, and [Hardy Weinberg Equilibrium P-value](https://en.wikipedia.org/wiki/Hardy–Weinberg_principle)). Notice how the histogram for HWE does not look as one would expect (most variants should have a p-value close to 1). This is because there are 5 populations represented in this dataset and the p-value we calculated includes all populations.
<img src="test.variantqc.png">

To compute the HWE p-value by population, we use a separate [`annotatevariants expr`](commands.html#annotatevariants_expr) command per population and the `hwe` function in the [Hail Expression Language](reference.html#HailExpressionLanguage) to compute HWE p-values. This function takes 3 integers representing the number of samples for each of the genotype categories (HomRef, Het, HomVar). 

To count the number of samples in each genotype category, we use a filter operation on an [aggregable of genotypes](index.html#aggregables). The genotypes are filtered by requiring the sample is from the population of interest and whether they are in the genotype class of interest. 
For example, to count the number of samples with European ancestry and are homozygotes for the reference allele, we filter genotypes where `sa.pheno.SuperPopulation == "EUR"` and the genotype call is homozygote reference `g.isHomRef`. 

After the filter function, we call the count function [`count()`](reference.html#aggregables) to count how many elements evaluated to True. The `.toInt` function is called to convert the output type of `count()`, which is a Long, to an Int which is required by the `hwe` function.
The results of [`printschema --va`](commands.html#printschema) shows we have added new fields in the variant annotations for HWE p-values for each population.

```
hail read -i test.filtersamples.vds \
    \
    annotatevariants expr -c 'va.hweByPop.hweEUR = 
                                hwe(gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomRef).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHet).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomVar).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweSAS = 
                                hwe(gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomRef).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHet).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomVar).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweAMR = 
                                hwe(gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomRef).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHet).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomVar).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweAFR = 
                                hwe(gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomRef).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHet).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomVar).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweEAS = 
                                hwe(gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomRef).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHet).count().toInt, 
                                    gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomVar).count().toInt)' \
    \
    printschema --va \
    \
    write -o test.hwebypop.vds    
```

<pre class="tutorial output" style="color: red">
Variant annotation schema:
va: Struct {
    hweByPop: Struct {
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
 
Now that the variant annotations contain a population-specific p-value for HWE, we can filter variants based on passing HWE in each population. The results of the [`count`](commands.html#count) command confirms that by calculating HWE p-values in each population separately, we only filter out 826 variants compared to 7098 variants before.

```
hail read -i test.hwebypop.vds \
    \
    filtervariants expr --keep -c 'va.hweByPop.hweEUR.pHWE > 1e-6 && 
                                   va.hweByPop.hweSAS.pHWE > 1e-6 && 
                                   va.hweByPop.hweAMR.pHWE > 1e-6 && 
                                   va.hweByPop.hweAFR.pHWE > 1e-6 && 
                                   va.hweByPop.hweEAS.pHWE > 1e-6' \
    \
    write -o test.hwefilter.vds \
    \
    count 
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants           10,135
</pre>

After creating a new VDS with population-specific annotations for HWE p-values, we add as annotations the results from the [`variantqc`](commands.html#variantqc) command.
We have to tell Hail where to find the variant name in the input text file. 
In this case, the components of the variant name are in different columns (Chrom, Pos, Ref, and Alt). 
To do this, we use the `-e` flag and specify we want to create a [`Variant`](reference.html#variant) object with the 4 corresponding column names from the file **test.variantqc.tsv** (Chrom, Pos, Ref, and Alt). 
However, because we used the `--impute` flag to impute the types of each column in the input file, the "Chrom" column is inferred to be an Integer. 
This creates a conflict because the Variant constructor expects the chromosome name to be a string.
Therefore, we use the `-t` flag to force the type of the "Chrom" column to be a String. 

Lastly we use the [`filtervariants expr`](commands.html#filtervariants_expr) command to keep variants with a mean GQ greater than or equal to 20 (a **boolean expression** designated by the `-c` flag) and then [`write`](commands.html#write) a new VDS where variants have been filtered for HWE and mean GQ (**test.filtervariants.vds**).


```
hail read -i test.hwefilter.vds \
    \
    annotatevariants table --impute -e 'Variant(Chrom, Pos, Ref, Alt)' \
        -r va.qc -t 'Chrom: String' test.variantqc.tsv \
    \
    filtervariants expr --keep -c 'va.qc.gqMean >= 20' \
    \
    write -o test.filtervariants.vds \
    \
    count 
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants            9,949
</pre>
  
Using the [`count`](commands.html#count) command, we can see we have filtered out 1,012 variants from the dataset.

#### Sex Check

One of the most important QC metrics is to ensure that the reported sex of the samples matches what the actual genetic sex is. 
A high sex check failure rate can indicate sample swaps may have occurred.

First, we count how many X chromosome variants are in the original dataset and find there are 273.

```
hail read -i test.filtersamples.vds \
    \
    filtervariants expr --keep -c 'v.contig == "X"' \
    \
    count
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants              273
</pre>
  
However, after variant QC, the number of X chromosome variants went from 273 to 10 (not enough for a sex check!)

```
hail read -i test.filtervariants.vds \
    \
    filtervariants expr --keep -c 'v.contig == "X"' \
    \
    count
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants               10
</pre>
  
The reason why this happened is the HWE p-values for the X chromosome should not include male samples in the calculation as they only have two possible genotypes (HomRef or HomVar). 
Therefore, we need to modify how we calculate HWE. 
We use a conditional expression so that when a variant is on the X chromosome, we only include female samples in the calculation.

```
hail read -i test.filtersamples.vds \
    \
    annotatevariants expr -c 'va.hweByPop.hweEUR = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EUR" && 
                                                        g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && 
                                                        g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && 
                                                        g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweSAS = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "SAS" && 
                                                        g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && 
                                                        g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && 
                                                        g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweAMR = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AMR" && 
                                                        g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && 
                                                        g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && 
                                                        g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweAFR = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AFR" && 
                                                        g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && 
                                                        g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && 
                                                        g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweEAS = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EAS" && 
                                                        g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && 
                                                        g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && 
                                                        g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    filtervariants expr --keep -c 'va.hweByPop.hweEUR.pHWE > 1e-6 && 
                                   va.hweByPop.hweSAS.pHWE > 1e-6 && 
                                   va.hweByPop.hweAMR.pHWE > 1e-6 && 
                                   va.hweByPop.hweAFR.pHWE > 1e-6 && 
                                   va.hweByPop.hweEAS.pHWE > 1e-6' \
    \
    annotatevariants table --impute -e 'Variant(Chrom, Pos, Ref, Alt)' \
        -r va.qc -t 'Chrom: String' test.variantqc.tsv \
    \
    filtervariants expr --keep -c 'va.qc.gqMean >= 20' \
    \
    write -o test.filtervariants2.vds \
    \
    count 
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants           10,160
</pre>

Now, when we count how many X chromosome variants there are, we get 221 variants.

```
hail read -i test.filtervariants2.vds \
    \
    filtervariants expr --keep -c 'v.contig == "X"' \
    \
    count
```

<pre class="tutorial output" style="color: red">
  nSamples             1,646
  nVariants              221
</pre>
  
To do a sex check, first we use the [`imputesex`](commands.html#imputesex) command with a minimum minor allele frequency threshold (`--maf-threshold`) of 0.05 to determine the genetic sex of a sample based on the inbreeding coefficient.
The [`imputesex`](commands.html#imputesex) command adds new sample annotations for whether a sample is a female to `sa.imputesex.isFemale`. 
We can then create a new sample annotation (`sa.sexcheck`) which compares whether the imputed sex (`sa.imputesex.isFemale`) is the same as the reported sex (`sa.pheno.isFemale`).

To output the sexcheck annotations to a text file, we use the [`exportsamples`](commands.html#exportsamples) command. 
The `-c` flag takes an expression that defines which columns should be output to a TSV file. 
Each column is defined by a field in sample annotations (example: `sa.pheno.isFemale`) or the sample (`s`). The column header is given by the label in front of the annotation.
For example, to output a column containing the reported sex of the sample, we assign the **column name** `ReportedSex` and assign it to the variable `sa.pheno.isFemale` from the sample annotations using an equals sign (`=`).
Multiple columns are separated by a comma. 
The `-o` flag defines the location of the output file (in this case **test.sexcheck.tsv**).

Lastly, to remove samples that failed the sex check above, we can use the [`filtersamples expr`](commands.html#filtersamples_expr) command to only keep (`--keep`) samples where the sample annotation `sa.sexcheck` evaluates to True.
In this case, we can filter samples based on the sample annotation field `sa.sexcheck` because it is a boolean variable.


```
hail read -i test.filtervariants2.vds \
    \
    imputesex --maf-threshold 0.05 \
    \
    annotatesamples expr -c 'sa.sexcheck = sa.pheno.isFemale == sa.imputesex.isFemale' \
    \
    exportsamples -c 'Sample = s, ReportedSex = sa.pheno.isFemale, 
                      ImputedSex = sa.imputesex.isFemale, SexAgree = sa.sexcheck' \
                  -o test.sexcheck.tsv \
    \
    filtersamples expr --keep -c 'sa.sexcheck' \
    \
    count
```

<pre class="tutorial output" style="color: red">
  nSamples             1,079
  nVariants           10,160
</pre>
  
We removed 567 samples where the genetic sex does not match the reported sex. This is an extremely high sex check failure rate! 
To figure out why this happened, we used the [`exportsamples`](commands.html#exportsamples) command to print out the sample annotations to a text file that we could analyze quickly with `awk`.

```
awk '{print $4}' test.sexcheck.tsv | sort | uniq -c
```

<pre class="tutorial output" style="color: red">
 564 NA
   1 SexAgree
   3 false
1079 true
</pre>

We found that the majority of the sex check failures were not true failures because the genetic sex could not be determined. There were 3 samples with sex check failures.
If we had more X-chromosome variants, we could get a better estimate of the inbreeding coefficient on the X chromosome and have more accurate sex check results.
To filter out samples where the imputed sex does not match the reported sex, we modify our [`filtersamples`](commands.html#filtersamples) expression to include samples where `sa.sexcheck` is True or `sa.sexcheck` is Undefined (`isMissing(sa.sexcheck)`).

```
hail read -i test.filtervariants2.vds \
    \
    imputesex --maf-threshold 0.05 \
    \
    annotatesamples expr -c 'sa.sexcheck = sa.pheno.isFemale == sa.imputesex.isFemale' \
    \
    exportsamples -c 'Sample = s, ReportedSex = sa.pheno.isFemale, 
                      ImputedSex = sa.imputesex.isFemale, SexAgree = sa.sexcheck' \
                  -o test.sexcheck.tsv \
    \
    filtersamples expr --keep -c 'sa.sexcheck || isMissing(sa.sexcheck)' \
    \
    write -o test.sexcheck.vds \
    \
    count    
```

<pre class="tutorial output" style="color: red">
  nSamples             1,643
  nVariants           10,160
</pre>

## PCA

To account for population stratification in association testing, we use principal component analysis to compute covariates that are proxies for genetic similarity.
For PCA to work, we need an independent set of SNPs. The text file **purcell5k.interval_list** contains a list of independent variants.
To calculate principal components, we first use the [`filtervariants intervals`](commands.html#filtervariants_intervals) command to only keep SNPs from the **purcell5k.interval_list**. Next, we use the [`pca`](commands.html#pca) command to calculate the first 10 principal components and output those to a text file.
Lastly, we export sample annotations so that we can plot the principal components and color the points by their population group.

```
hail read -i test.sexcheck.vds \
    \
    filtervariants intervals --keep -i $prunedVariants \
    \
    pca -o test.pca.tsv \
    \
    annotatesamples table -e Sample -r sa.pca -i test.pca.tsv --impute \
    \
    exportsamples -c 'Sample = s, SuperPopulation = sa.pheno.SuperPopulation, 
                      Population = sa.pheno.Population, PC1 = sa.pca.PC1, 
                      PC2 = sa.pca.PC2, PC3 = sa.pca.PC3' \
                  -o test.pcaPlusPop.tsv
```

Here are some examples plotted using R:
<img src="test.pcaPlot.png">

Lastly, we add the principal components computed above to the QC'd dataset where unreliable genotypes, samples, and variants have been filtered out.
We have printed out all three schemas (global, sample annotations, variant annotations) so you can see all of the annotations we have added to the dataset. 

```
hail read -i test.sexcheck.vds \
    \
    annotatesamples table -e Sample -r sa.pca -i test.pca.tsv --impute \
    \
    write -o test.qc.vds \
    \
    printschema
```

<pre class="tutorial output" style="color: red">
Global annotation schema:
global: Empty

Sample annotation schema:
sa: Struct {
    pheno: Struct {
        Sample: String,
        Population: String,
        SuperPopulation: String,
        isFemale: Boolean,
        PurpleHair: Boolean,
        CaffeineConsumption: Double
    },
    qc: Struct {
        Sample: String,
        callRate: Double,
        nCalled: Int,
        nNotCalled: Int,
        nHomRef: Int,
        nHet: Int,
        nHomVar: Int,
        nSNP: Int,
        nInsertion: Int,
        nDeletion: Int,
        nSingleton: Int,
        nTransition: Int,
        nTransversion: Int,
        dpMean: Double,
        dpStDev: Double,
        gqMean: Double,
        gqStDev: Double,
        nNonRef: Int,
        rTiTv: Double,
        rHetHomVar: Double,
        rDeletionInsertion: String
    },
    imputesex: Struct {
        isFemale: Boolean,
        Fstat: Double,
        nTotal: Int,
        nCalled: Int,
        expectedHoms: Double,
        observedHoms: Int
    },
    sexcheck: Boolean,
    pca: Struct {
        Sample: String,
        PC1: Double,
        PC2: Double,
        PC3: Double,
        PC4: Double,
        PC5: Double,
        PC6: Double,
        PC7: Double,
        PC8: Double,
        PC9: Double,
        PC10: Double
    }
}

Variant annotation schema:
va: Struct {
    rsid: String,
    qual: Double,
    filters: Set[String],
    pass: Boolean,
    info: Struct {
        AC: Array[Int],
        AF: Array[Double],
        AN: Int,
        BaseQRankSum: Double,
        ClippingRankSum: Double,
        DP: Int,
        DS: Boolean,
        FS: Double,
        HaplotypeScore: Double,
        InbreedingCoeff: Double,
        MLEAC: Array[Int],
        MLEAF: Array[Double],
        MQ: Double,
        MQ0: Int,
        MQRankSum: Double,
        QD: Double,
        ReadPosRankSum: Double,
        set: String
    },
    aIndex: Int,
    wasSplit: Boolean,
    hweByPop: Struct {
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
    },
    qc: Struct {
        Chrom: String,
        Pos: Int,
        Ref: String,
        Alt: String,
        callRate: Double,
        AC: Int,
        AF: Double,
        nCalled: Int,
        nNotCalled: Int,
        nHomRef: Int,
        nHet: Int,
        nHomVar: Int,
        dpMean: Double,
        dpStDev: Double,
        gqMean: Double,
        gqStDev: Double,
        nNonRef: Int,
        rHeterozygosity: Double,
        rHetHomVar: Double,
        rExpectedHetFrequency: Double,
        pHWE: Double
    }
}
</pre>

## Association Testing

Now that we have a QC'd dataset with principal components calculated and phenotype information added, we can test for an association between the genetic variants and the phenotypes of PurpleHair (Dichotomous) and CaffeineConsumption (Continuous).

#### Linear Regression with Covariates

First, we [`read`](commands.html#read) the data from the QC'd VDS we created above. Next, we filter out variants with a minor allele frequency less than 5% (also include 95% as it's possible for the minor allele to be the reference allele).
Next, we use the [`linreg`](commands.html#linreg) command and specify the response variable `-y` to be the sample annotation for CaffeineConsumption `sa.pheno.CaffeineConsumption`. We also define 4 covariates to correct for: `sa.pca.PC1`, `sa.pca.PC2`, `sa.pca.PC3`, `sa.pheno.isFemale`.
The results of the [`linreg`](commands.html#linreg) command are put into the variant annotations and can be accessed with the root name `va.linreg`.
Lastly we use the [`exportvariants`](commands.html#exportvariants) command to export the results of the linear regression to a text file for making a Q-Q plot in R.

```
hail read -i test.qc.vds \
    \
    filtervariants expr --keep -c 'va.qc.AF > 0.05 && va.qc.AF < 0.95' \
    \
    linreg -y sa.pheno.CaffeineConsumption -c 'sa.pca.PC1, sa.pca.PC2, 
                                               sa.pca.PC3, sa.pheno.isFemale' \
    \
    exportvariants -c 'Variant = v, Beta = va.linreg.beta, 
                       SE = va.linreg.se, T = va.linreg.tstat, 
                       PVAL = va.linreg.pval' \
                   -o test.linreg.tsv
```

<img src="test.linreg.qq.png">


#### Logistic Regression with Covariates

First, we [`read`](commands.html#read) the data from the QC'd VDS we created above. Next, we filter out variants with a minor allele frequency less than 5% (also include 95% as it's possible for the minor allele to be the reference allele).
Next, we use the [`logreg`](commands.html#logreg) command and specify the response variable `-y` to be the sample annotation for PurpleHair `sa.pheno.PurpleHair`. We also define 4 covariates to correct for: `sa.pca.PC1`, `sa.pca.PC2`, `sa.pca.PC3`, `sa.pheno.isFemale`.
The results of the [`logreg`](commands.html#logreg) command are put into the variant annotations and can be accessed with the root name `va.logreg`.
Lastly we use the [`exportvariants`](commands.html#exportvariants) command to export the results of the logistic regression to a text file for making a Q-Q plot in R.

```
hail read -i test.qc.vds \
    \
    filtervariants expr --keep -c 'va.qc.AF > 0.05 && va.qc.AF < 0.95' \
    \
    logreg -y 'sa.pheno.PurpleHair' -t wald -c 'sa.pca.PC1, sa.pca.PC2, 
                                                sa.pca.PC3, sa.pheno.isFemale' \
    \
    exportvariants -c 'Variant = v, PVAL = va.logreg.wald.pval' \
                   -o test.logreg.tsv
```

<img src="test.logreg.qq.png">


#### Fisher's Exact Test for Rare Variants

First, we [`read`](commands.html#read) the data from the QC'd VDS we created above. Next, we filter out variants with a minor allele frequency greater than 5% and less than 95%, so we're left with rare variants.
Next we perform 4 annotate variant commands using [genotype aggregables](reference.html#aggregables) to count the number of minor alleles and major alleles per phenotype status.
These new variant annotations can be used as inputs to the `fet` function which takes 4 integers representing a 2x2 contingency table. We define the output of the [`fet`](reference.html#fet) function will go into the variant annotations keyed by `va.fet`.
Lastly, we export the results to a text file and make a Q-Q plot in R. 

```
hail read -i test.qc.vds \
    \
    filtervariants expr --keep -c 'va.qc.AF <= 0.05 && va.qc.AF >= 0.95' \
    \
    annotatevariants expr -c 'va.minorCase = 
                                gs.filter(g => sa.pheno.PurpleHair && g.isHet).count() +
                                2 * gs.filter(g => sa.pheno.PurpleHair && g.isHomVar).count()' \
    \
    annotatevariants expr -c 'va.minorControl = 
                                gs.filter(g => !sa.pheno.PurpleHair && g.isHet).count() + 
                                2 * gs.filter(g => !sa.pheno.PurpleHair && g.isHomVar).count()' \
    \
    annotatevariants expr -c 'va.majorCase = 
                                gs.filter(g => sa.pheno.PurpleHair && g.isHet).count() +
                                2 * gs.filter(g => sa.pheno.PurpleHair && g.isHomRef).count()' \
    \
    annotatevariants expr -c 'va.majorControl = 
                                gs.filter(g => !sa.pheno.PurpleHair && g.isHet).count() +
                                2 * gs.filter(g => !sa.pheno.PurpleHair && g.isHomRef).count()' \
    \
    annotatevariants expr -c 'va.fet = 
                                fet(va.minorCase.toInt, va.minorControl.toInt, 
                                    va.majorCase.toInt, va.majorControl.toInt)' \
    \
    exportvariants -c 'Variant = v, MinorCase = va.minorCase, MinorControl = va.minorControl, 
                       MajorCase = va.majorCase, majorControl = va.majorControl, PVAL = va.fet.pValue, 
                       OR = va.fet.oddsRatio, ciLower = va.fet.ci95Lower, ciUpper = va.fet.ci95Upper' \
                   -o test.fet.tsv
```

<img src="test.fet.qq.png">

### Summary
 
Most concise way to write this analysis:

```
hail importvcf $vcf \
    \
    splitmulti \
    \
    annotatesamples table --root sa.pheno -e Sample \
        --types 'Population: String, SuperPopulation: String, isFemale: Boolean, 
                 PurpleHair: Boolean, CaffeineConsumption: Double' 
        --input $sampleAnnotations \
    \
    filtergenotypes --keep -c 'let ab = g.ad[1] / g.ad.sum 
                               in ((g.isHomRef && ab <= 0.1) || 
                                   (g.isHet && ab >= 0.25 && ab <= 0.75) || 
                                   (g.isHomVar && ab >= 0.9))' \
    \
    write -o test.filtergeno.vds    


hail read -i test.filtergeno.vds \
    \
    filtervariants expr --keep -c 'let callRate = gs.filter(g => g.isCalled).count() / gs.count() 
                                   in callRate >= 0.95' \
    \
    sampleqc -o test.sampleqc.tsv


hail read -i test.filtergeno.vds \
    \
    annotatesamples table -e Sample -r sa.qc --impute -i test.sampleqc.tsv \
    \
    filtersamples expr --keep -c 'sa.qc.callRate >= 0.97 && sa.qc.gqMean >= 20' \
    \
    write -o test.filtersamples.vds


hail read -i test.filtersamples.vds \
    \
    variantqc -o test.variantqc.tsv \
    \
    annotatevariants expr -c 'va.hweByPop.hweEUR = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EUR" && g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweSAS = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "SAS" && g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweAMR = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AMR" && g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweAFR = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "AFR" && g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    annotatevariants expr -c 'va.hweByPop.hweEAS = 
                                if (v.contig != "X") 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomRef).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHet).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomVar).count().toInt) 
                                else 
                                    hwe(gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomRef && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHet && sa.pheno.isFemale).count().toInt, 
                                        gs.filter(g => sa.pheno.SuperPopulation == "EAS" && g.isHomVar && sa.pheno.isFemale).count().toInt)' \
    \
    filtervariants expr --keep -c 'va.hweByPop.hweEUR.pHWE > 1e-6 && 
                                   va.hweByPop.hweSAS.pHWE > 1e-6 && 
                                   va.hweByPop.hweAMR.pHWE > 1e-6 && 
                                   va.hweByPop.hweAFR.pHWE > 1e-6 && 
                                   va.hweByPop.hweEAS.pHWE > 1e-6' \
    \
    filtervariants expr --keep -c 'va.qc.gqMean >= 20' \
    \
    write -o test.filtervariants.vds


hail read -i test.filtervariants.vds \
    \
    imputesex --maf-threshold 0.05 \
    \
    annotatesamples expr -c 'sa.sexcheck = sa.pheno.isFemale == sa.imputesex.isFemale' \
    \
    exportsamples -c 'Sample = s, ReportedSex = sa.pheno.isFemale, 
                      ImputedSex = sa.imputesex.isFemale, SexAgree = sa.sexcheck' \
                  -o test.sexcheck.tsv \
    \
    filtersamples expr --keep -c 'sa.sexcheck || isMissing(sa.sexcheck)' \
    \
    write -o test.sexcheck.vds


hail read -i test.sexcheck.vds \
    \
    filtervariants intervals --keep -i $prunedVariants \
    \
    pca -o test.pca.tsv


hail read -i test.sexcheck.vds \
    \
    annotatesamples table -e Sample -r sa.pca -i test.pca.tsv --impute \
    \
    write -o test.qc.vds


hail read -i test.qc.vds \
    \
    filtervariants expr --keep -c 'va.qc.AF > 0.05' \
    \
    linreg -y sa.pheno.CaffeineConsumption -c 'sa.pca.PC1, sa.pca.PC2, sa.pca.PC3, sa.pheno.isFemale' \
    \
    exportvariants -c 'Variant = v, Beta = va.linreg.beta, SE = va.linreg.se, 
                       T = va.linreg.tstat, PVAL = va.linreg.pval' \
                   -o test.linreg.tsv


hail read -i test.qc.vds \
    \
    filtervariants expr --keep -c 'va.qc.AF > 0.05' \
    \
    logreg -y 'sa.pheno.PurpleHair' -t wald -c 'sa.pca.PC1, sa.pca.PC2, sa.pca.PC3, sa.pheno.isFemale' \
    \
    exportvariants -c 'Variant = v, PVAL = va.logreg.wald.pval' -o test.logreg.tsv


hail read -i test.qc.vds \
    \
    filtervariants expr --keep -c 'va.qc.AF <= 0.05' \
    \
    annotatevariants expr -c 'va.minorCase = 
                                gs.filter(g => sa.pheno.PurpleHair && g.isHet).count() +
                                2 * gs.filter(g => sa.pheno.PurpleHair && g.isHomVar).count()' \
    \
    annotatevariants expr -c 'va.minorControl = 
                                gs.filter(g => !sa.pheno.PurpleHair && g.isHet).count() + 
                                2 * gs.filter(g => !sa.pheno.PurpleHair && g.isHomVar).count()' \
    \
    annotatevariants expr -c 'va.majorCase = 
                                gs.filter(g => sa.pheno.PurpleHair && g.isHet).count() +
                                2 * gs.filter(g => sa.pheno.PurpleHair && g.isHomRef).count()' \
    \
    annotatevariants expr -c 'va.majorControl = 
                                gs.filter(g => !sa.pheno.PurpleHair && g.isHet).count() +
                                2 * gs.filter(g => !sa.pheno.PurpleHair && g.isHomRef).count()' \
    \
    annotatevariants expr -c 'va.fet = 
                                fet(va.minorCase.toInt, va.minorControl.toInt, 
                                    va.majorCase.toInt, va.majorControl.toInt)' \
    \
    exportvariants -c 'Variant = v, MinorCase = va.minorCase, MinorControl = va.minorControl, 
                       MajorCase = va.majorCase, majorControl = va.majorControl, PVAL = va.fet.pValue, 
                       OR = va.fet.oddsRatio, ciLower = va.fet.ci95Lower, ciUpper = va.fet.ci95Upper' \
                   -o test.fet.tsv
```

