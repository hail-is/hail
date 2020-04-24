<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
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
                <h1 id='logo-title'>Powering biobank-scale genetics</h1>
                <subtitle id="logo-subtitle">An open-source language for bioinformatics that unifies your data</subtitle>
                <div style="display: flex;" id="hero-button-container">
                    <a class="button" href='/docs/0.2'>Learn</a>
                    <a class="button" href='https://hail.zulipchat.com' target='_blank'>Chat with us</a>
                </div>
            </div>
        </div>
        <div id="about" class="about dark">
            <div class="header-wrap">
                <h1>About</h1>
            </div>
            <div class="about-content">
                <p> Hail is an open-source library that allows for the efficient exploration of genomic data.
                    Successfully, it can be used to interrogate <b>biobank scale</b> genomic data such as the
                    UK Biobank, TopMed, FinnGen, and Biobank Japan.
                    We want to bring the idea of data frames to genetics by driving <b>modern data scaling</b>:
                </p>
                <p> Modern data science stack is driven by data frames where modern data science tools, e.g. Python:pandas
                    and R:dplyr, for genomic data lack the data frame scalability offered by Hail.
                    We want to bring efficient use of data frames to genomic analysis by offering the relatively effortless
                    manipulation offered by the use of Hail <a href='/docs/0.2/overview/matrix_table.html?highlight=matrix%20table' target='_blank'>MatrixTables</a>.
                </p>
                <p>
                    The Hail MatrixTable is a data frame abstraction that transcends data formats. Therefore, a MatrixTable is a
                    format that allows for unified input formats e.g. vcf, bgen, tsv, gtf, bed files. Upon creating the unified
                    input formats, Hail is a unified analytics platform which we want our users to feel comfortable in using to
                    empower their science
                </p>
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
                <p>We would like to thank 
                    <a href='https://zulipchat.com/' target='_blank'>Zulip</a> for supporting open-source
                    by providing free hosting, and YourKit, LLC for generously providing free licenses for 
                    <a href='https://www.yourkit.com/java/profiler/'>YourKit Java Profiler</a> for open-source development.
                </p>
            </div>
        </div>
        <script src="/vendors/vanta/threeR115.min.js"></script>
        <script type="module">
            <![CDATA[
                import Viz from "/vendors/vanta/viz.js";
                new Viz({
                    el: "#hero-background",
                    mouseControls: false,
                    touchControls: false,
                    minHeight: 200.0,
                    minWidth: 200.0,
                    scale: 1.0,
                    scaleMobile: 1.0,
                    points: 11,
                    maxDistance: 20,
                    spacing: 15,
                    backgroundColor: "#fff",
                    color: "#283870",
                });
            ]]>
        </script>
    </xsl:template>
</xsl:stylesheet>
