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
    <span id='home'>
        <div id="hero">
            <div id="hero-background"></div>
            <div id="all-hero-content">
                <div id="hero-content">
                    <h1 id="logo-title">Powering genomic analysis, at every scale</h1>
                    <subtitle class="logo-subtitle">An open-source library for scalable genomic data exploration</subtitle>
                    <div style="display: flex;" id="hero-button-container">
                        <a class="button" href="#install">Install</a>
                        <a class="button" href="#features">Features</a>
                        <a class="button" href="/gethelp.html">Get Help</a>
                    </div>
                </div>

                <div id="hero-content-right">
                    <pre class="right language-python" style="display:flex;min-width: 500px;">
                        <img id="reveal-img" src="/hail-tutorial-gwas-plot-cropped-opt.png" style="cursor: pointer;width:600px;height:360px;" onclick="reveal(this);" />
                        <code id="reveal-code" class="language-python" style="display:none;width:600px;height:360px; cursor: pointer;" onclick="hide(this);">import hail as hl

mt = hl.read_matrix_table('resources/post_qc.mt')
mt = mt.filter_rows(hl.agg.call_stats(mt.GT, mt.alleles).AF[1] > 0.01)
pca_scores = hl.hwe_normalized_pca(mt.GT, k = 5, True)[1]
mt = mt.annotate_cols(pca = pca_scores[mt.s])

gwas = hl.linear_regression_rows(
       y=mt.pheno.caffeine_consumption, 
       x=mt.GT.n_alt_alleles(),
       covariates=[1.0, mt.pheno.is_female,
                   mt.pca.scores[0], mt.pca.scores[1],
                   mt.pca.scores[2]])

p = hl.plot.manhattan(gwas.p_value)
show(p)
</code>
                    </pre>

                    <subtitle class="logo-subtitle small">GWAS with Hail (click to <span id='reveal-text'>show code</span>)</subtitle>
                </div>
            </div>
        </div>
        <div id="install" class="about dark" style="z-index:1">
            <div class="header-wrap" style="justify-content: center;">
                <h1>Install</h1>
            </div>
            <div class="about-content columns" style="align-self: center;">
                <section style="flex-direction:column;margin:auto;">
                    <div style="background:white;padding:20px; color:black;text-align:center;margin:20px 0px;display:block;font-family:'Courier New', Courier, monospace">
                    pip install hail</div>
                    <p>
                        Hail requires Python 3 and the
                        <a href="https://adoptopenjdk.net/index.html" target="_blank">Java 8
                        JRE</a>
                    </p>
                    <p>GNU/Linux wil also need C and C++ standard libraries if not already installed</p>
                    <p>
                        <a href="/docs/0.2/getting_started.html">Detailed instructions</a>
                    </p>
                </section>
            </div>
        </div>
        <div id="features" class="about">
            <div class="content">
                <div class="header-wrap" styel="justify-content: space-between">
                    <h1>Features</h1>
                </div>
                <div class="about-content columns">
                    <section>
                        <h4>Simplified Analysis</h4>
                        <p>Hail is an open-source Python library that simplifies genomic data analysis.
                        It provides powerful, easy-to-use data science tools that can be used to interrogate even
                        biobank-scale genomic data (e.g UK Biobank, TopMed, FinnGen, and Biobank Japan).
                    </p>
                    </section>
                    <section>
                        <h4>Genomic Dataframes</h4>
                        <p>Modern data science is driven by table-like data structures, often called dataframes (see <a href="https://pandas.pydata.org">Pandas</a>).
                            While convenient, they don't capture the structure of genetic data, which has row (variant) and
                            column
                            (genotype) groups.
                            To remedy this, Hail introduces a distributed, dataframe-like structure called
                            <a href="/docs/0.2/overview/matrix_table.html?highlight=matrix%20table" target="_blank">MatrixTable</a>.
                        </p>
                    </section>
                    <section>
                        <h4>Input Unification</h4>
                        <p>The <a href="/docs/0.2/overview/matrix_table.html?highlight=matrix%20table" target="_blank">Hail
                            MatrixTable</a> unifies a wide range of input formats (e.g. vcf, bgen, plink, tsv, gtf, bed files),
                            and supports scalable queries, even on petabyte-size datasets.
                            By leveraging MatrixTable, Hail provides an integrated, scalable analysis platform for science.
                        </p>
                    </section>
                </div>
                <a class="button" href="/tutorial.html" style='align-self:flex-end; margin-top:1rem;'>Learn More ></a>
            </div>
        </div>
        <div class="about dark">
            <div class="content">
                <div class="header-wrap">
                    <h1>Acknowledgments</h1>
                </div>
                <div class="about-content">
                    <p>The Hail team has several sources of funding at the Broad Institute:</p>
                    <ul>
                        <li>
                        The Stanley Center for Psychiatric Research, which together with
                        Neale Lab has provided an incredibly supportive and stimulating
                        home.
                    </li>
                        <li>
                        Principal Investigator Benjamin Neale, whose
                        scientific leadership has been essential for solving the right
                        problems.
                    </li>
                        <li>
                        Principal Investigator Daniel MacArthur and the other members 
                        of the gnomAD council.
                    </li>
                        <li>
                        Jeremy Wertheimer, whose strategic advice and generous
                        philanthropy have been essential for growing the impact of Hail.
                    </li>
                    </ul>
                    <p>We are grateful for generous support from:</p>
                    <ul>
                        <li>
                        The National Institute of Diabetes and Digestive and Kidney
                        Diseases
                    </li>
                        <li>The National Institute of Mental Health</li>
                        <li>The National Human Genome Research Institute</li>

                    </ul>
                    <p>We are grateful for generous past support from:</p>
                    <ul>
                        <li>The Chan Zuckerburg Initiative</li>
                    </ul>
                    <p>
                        We would like to thank
                        <a href="https://zulipchat.com/" target="_blank">Zulip</a>
                        for supporting open-source
                    by providing free hosting, and YourKit, LLC for generously providing free licenses for
                        <a href="https://www.yourkit.com/java/profiler/">YourKit Java Profiler</a>
                        for open-source
                    development.
                    </p>
                </div>
            </div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.20.0/components/prism-core.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.20.0/components/prism-python.min.js"></script>
        <script>
            <![CDATA[
                const cachedImg = document.getElementById("reveal-img");
                const cachedCode = document.getElementById("reveal-code");
                const cachedText = document.getElementById("reveal-text");
                const defText = cachedText.textContent;

                function reveal(e) {
                    cachedCode.style.display = "block";
                    cachedImg.style.display = "none";
                    cachedText.textContent = "hide code";
                }
                function hide(e) {
                    cachedCode.style.display = "none";
                    cachedImg.style.display = "initial";
                    cachedText.textContent = defText;
                }
            ]]>
        </script>
        <script src="/vendors/vanta/threeR115.min.js" async="true"></script>
        <script type="module" crossorigin="use-credentials"> <!-- https://github.com/hail-is/hail/pull/8928#issuecomment-639218007 -->
            <![CDATA[
                function checkViz() {
                    if(window.THREE) {
                        import("/vendors/vanta/viz.min.js").then(module => {
                            new module.default({
                                el: "#hero-background",
                                points: 10,
                                maxDistance: 23,
                                spacing: 20,
                                backgroundColor: "#fff",
                                color: "#283870",
                            });
                        })
                        return;
                    }

                    setTimeout(checkViz, 16);
                }

                checkViz();
            ]]>
        </script>
    </span>
    </xsl:template>
</xsl:stylesheet>