<?xml version="1.0" encoding="ISO-8859-15" ?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:import href="template.xslt" />
    <xsl:template match="h1[@id='hail']"></xsl:template>
    <xsl:template name="page-title">Hail</xsl:template>
    <xsl:template name="meta-description">
        <meta name="description" content="Hail Overview" />
    </xsl:template>
    <xsl:template name="header">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.20.0/themes/prism.min.css"></link>
    </xsl:template>
    <xsl:template match="body">
        <div id="hero" class='dark short'>
            <div id="hero-content" class='wide'>
                <h1 id="logo-title">Import, prototype, scale
                </h1>
                <div class="logo-subtitle">Perform analyses with distributed
                    dataframe-like
                    collections</div>

            </div>
        </div>
        <div id="about" class="about staggered">
            <div class='content'>
                <section class='left'>
                    <div class='el'>
                        <pre class='left'><code class="language-python">import hail as hl
mt = hl.import_vcf('gs://bucket/path/myVCF.vcf.bgz')
mt.write('gs://bucket/path/dataset.mt', overwrite=True)
# read matrix into env
mt = hl.read_matrix_table('gs://bucket/path/dataset.mt')
mt1 = hl.import_vcf('/path/to/my.vcf.bgz')
mt2 = hl.import_bgen('/path/to/my.bgen')
mt3 = hl.import_plink(bed='/path/to/my.bed',
                      bim='/path/to/my.bim',
                      fam='/path/to/my.fam')</code></pre>
                    </div>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Input Unification
                        </h3>
                        <p>Import formats such as bed, bgen, plink, or vcf, and manipulate them using a common dataframe-like interface.</p>
                    </div>

                </section>
                <svg>
                    <path d="M 250 200 L 250 300 L 400 300 L 400 300 L 450 300 L 650 300 L 650 400 " fill="transparent" />
                </svg>
                <section class='right'>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Genomic Dataframes</h3>
                        <p>For large and dense structured matrices, like sequencing data, coordinate representations are
                            both
                            hard to work with and computationally inefficient. A core piece of Hail functionality is the
                            MatrixTable, a 2-dimensional generalization of Table. The MatrixTable makes it possible to
                            filter,
                            annotate, and aggregate symmetrically over rows and columns.</p>
                    </div>
                    <div class='el'>
                        <pre class='right'><code class="language-python"># What is a MatrixTable?
mt.describe(widget=True)

# filter to rare, loss-of-function variants
mt = mt.filter_rows(mt.variant_qc.AF[1] &lt; 0.005)
mt = mt.filter_rows(mt.csq == 'LOF')
</code></pre>
                    </div>

                </section>
                <svg>
                    <path d="M 250 400 L 250 300 L 400 300 L 400 300 L 450 300 L 650 300 L 650 200 " fill="transparent" />
                </svg>
                <section class='left'>
                    <div class='el'>
                        <pre class='left'><code class="language-python"># run sample QC and save into matrix table
mt = hl.sample_qc(mt)

# filter for samples that are > 95% call rate
mt = mt.filter_cols(mt.sample_qc.call_rate >= 0.95) 

# run variant QC and save into matrix table
mt = hl.variant_qc(mt)

# filter for variants that are >95% call rate and >1% frequency
mt = mt.filter_rows(mt.variant_qc.call_rate > 0.95)
mt = mt.filter_rows(mt.variant_qc_.AF[1] > 0.01)</code></pre>
                    </div>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Simplified Analysis</h3>
                        <p>Hail makes it easy to analyze your data. Let's start by filtering a dataset by variant and sample
                            quality metrics, like call rate and allele frequency.</p>
                    </div>
                </section>
                <svg>
                    <path d="M 250 200 L 250 300 L 400 300 L 400 300 L 450 300 L 650 300 L 650 400 " fill="transparent" />
                </svg>
                <section class='right'>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Quality Control Procedures</h3>
                        <p>Quality control procedures, like sex check, are made easy using Hail's declarative syntax</p>
                    </div>
                    <div class='el'>
                        <pre class='right'><code class="language-python">imputed_sex = hl.impute_sex(mt.GT)
mt = mt.annotate_cols(
     sex_check = imputed_sex[mt.s].is_female == mt.reported_female)</code></pre>
                    </div>
                </section>
                <svg>
                    <path d="M 250 400 L 250 300 L 400 300 L 400 300 L 450 300 L 650 300 L 650 200 " fill="transparent" />
                    <!--d="M8,103 C-2,468 499,78 491,463"-->
                    <!--curved: M 100 200 Q 100 300 200 300 L 300 300 L 400 300 L 500 300 L 600 300 Q 700 300 700 400-->
                </svg>
                <section class='left'>
                    <div class='el'>
                        <pre class='left'><code class="language-python"># must use Google cloud platform for this to work 
# annotation with vep
mt = hl.vep(mt)</code></pre>
                    </div>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Variant Effect Predictor</h3>
                        <p>Annotating variants with Variant effect predictor has never been easier.</p>
                    </div>
                </section>

                <svg>
                    <path d="M 250 400 L 250 300 L 400 300 L 400 300 L 450 300 L 650 300 L 650 200 " fill="transparent" />
                </svg>
                <section class='right'>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Rare-Variant Association Testing</h3>
                        <p>Perform Gene Burden Tests on sequencing data with just a few lines of Python.</p>
                    </div>
                    <div class='el'>
                        <pre class='right'><code class="language-python">gene_intervals = hl.read_table("gs://my_bucket/gene_intervals.t")
mt = mt.annotate_rows(gene = gene_intervals.index(mt.locus,
                             all_matches=True).gene_name)

mt = mt.explode_rows(mt.gene)
mt = mt.group_rows_by(mt.gene)
        .aggregate(burden = hl.agg.count_where(mt.GT.is_non_ref()))

result = hl.linear_regression_rows(y=mt.phenotype, x=mt.burden)</code></pre>
                    </div>
                </section>
                <svg>
                    <path d="M 250 400 L 250 300 L 400 300 L 400 300 L 450 300 L 650 300 L 650 200 " fill="transparent" />
                </svg>
                <section class='left'>
                    <div class='el'>
                        <pre class='left'><code class="language-python"># generate and save PC scores
eigenvalues, pca_scores, _ = hl.hwe_normalized_pca(mt.GT, k=4)


# run linear regression for the first 4 PCs
mt = mt.annotate_cols(scores = pca_scores[mt.sample_id].scores)
results = hl.linear_regression_rows(
    y=mt.phenotype,
    x=mt.GT.n_alt_alleles(),
    covariates=[1, mt.scores[0], mt.scores[1],
                mt.scores[2], mt.scores[3]]
)</code></pre>
                    </div>
                    <div class='el'>
                        <h3 style='margin-top:0px'>Principal Component Analysis (PCA)</h3>
                        <p>Adjusting GWAS models with principal components as covariates has never been easier.</p>
                    </div>
                </section>
            </div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.20.0/components/prism-core.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.20.0/components/prism-python.min.js"></script>
    </xsl:template>
</xsl:stylesheet>