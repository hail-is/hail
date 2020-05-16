<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:import href="template.xslt"/>
    <xsl:template match="h1[@id='hail']"></xsl:template>
    <xsl:template name="page-title">Hail</xsl:template>
    <xsl:template name="meta-description">
        <meta name="description" content="Hail Overview"/>
    </xsl:template>
    <xsl:template match="body">
        <div id="hero">
            <div id="hero-background"></div>
            <div id="hero-content">
                <h1 id="logo-title">Powering biobank-scale genomics</h1>
                <subtitle id="logo-subtitle">An open-source library for scalable genomic data exploration</subtitle>
                <div style="display: flex;" id="hero-button-container">
                    <a class="button" href="/docs/0.2/getting_started.html#installation">Install</a><a class="button" href="#features">Features</a>
                </div>
            </div>
        </div>
        <div id="features" class="about dark">
            <div class="header-wrap" styel="justify-content: space-between">
                <h1>Features</h1><a class="button" href="/tutorial.html" style='align-self: flex-end;'>Learn More ></a>
            </div>
            <div class="about-content columns">
                <section>
                    <h4>Simplified Analysis</h4>
                    <p> Hail is an open-source Python library that simplifies genomic data analysis.
                        It provides powerful, easy-to-use data science tools that can be used to interrogate even
                        biobank-scale genomic data (e.g UK Biobank, TopMed, FinnGen, and Biobank Japan).
                    </p>
                </section>
                <section>
                    <h4>Genomic Dataframes</h4>
                    Modern data science is driven by table-like data structures, often called dataframes (see
                    <a href="https://pandas.pydata.org">Pandas</a>).
                    While convenient, they don't capture the structure of genetic data, which has row (variant) and column
                    (genotype) groups.
                    To remedy this, Hail introduces a distributed, dataframe-like structure called
                    <a href="/docs/0.2/overview/matrix_table.html?highlight=matrix%20table" target="_blank">MatrixTable</a>.
                </section>
                <section>
                    <h4>Input Unification</h4>
                    The <a href="/docs/0.2/overview/matrix_table.html?highlight=matrix%20table" target="_blank">Hail
                        MatrixTable</a> unifies a wide range of input formats (e.g. vcf, bgen, plink, tsv, gtf, bed files),
                    and supports scalable queries, even on petabyte-size datasets.
                    By leveraging MatrixTable, Hail provides an integrated, scalable analysis platform for science.
                </section>
            </div>
        </div>
        <div class="about">
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
                        Principal Investigators Benjamin Neale and Daniel MacArthur, whose
                        scientific leadership has been essential for solving the right
                        problems.
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
                    <li>The Chan Zuckerburg Initiative</li>
                </ul>
                <p>
                    We would like to thank
                    <a href="https://zulipchat.com/" target="_blank">Zulip</a> for supporting open-source
                    by providing free hosting, and YourKit, LLC for generously providing free licenses for
                    <a href="https://www.yourkit.com/java/profiler/">YourKit Java Profiler</a> for open-source development.
                </p>
            </div>
        </div>
        <script src="/vendors/vanta/threeR115.min.js"></script>
        <script type="module">
            <![CDATA[
                import Viz from "/vendors/vanta/viz.min.js";
                new Viz({
                    el: "#hero-background",
                    points: 10,
                    maxDistance: 23,
                    spacing: 20,
                    backgroundColor: "#fff",
                    color: "#283870",
                });
            ]]>
        </script>
    </xsl:template>
</xsl:stylesheet>
