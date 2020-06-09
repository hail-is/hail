<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:import href="template.xslt"/>
    <xsl:template match="h1[@id='hail']"></xsl:template>
    <xsl:template name="page-title">Hail</xsl:template>
    <xsl:template name="meta-description">
        <meta name="description" content="Hail Overview"/>
    </xsl:template>
    <xsl:template name="header">
    </xsl:template>
    <xsl:template match="body">
        <span id='gethelp' class='non-home'>
            <div id="hero" style='height:calc(25vh - 50px); min-height: 250px; background: #2a3a8c; color: #fff'>
                <div id="hero-content" style='box-shadow: none; background: inherit' class='wide'>
                    <h1 id="logo-title" style='background: inherit; color: inherit;animation: none'>Get Help!</h1>
                    <subtitle id="logo-subtitle" style='background:inherit;animation: none; opacity: 1'>Let us assist you on
                your journey to efficient genomic analysis</subtitle>
                </div>
            </div>
            <div id="about" class="about">
                <section class='left'>
                    <a class='button' href='/cheatsheets.html'>Cheatsheets</a>
                    <div>Cheatsheets are two-page PDFs loaded with short examples and even shorter explanations. They push you over all the little roadblocks.</div>
                </section>
                <section class='left'>
                    <a class='button' href='/docs/0.2/index.html'>Docs</a>
                    <div>When you need to find detailed information on how to get started with Hail, examples of Hail use, and how a function works: the reference document is your go to. To do a quick search of a Hail function, try out the search bar in the documentation.</div>
                </section>
                <section class='left'>
                    <a class='button' href='https://discuss.hail.is/new-topic?category=Help%20%5B0.2%5D'>Ask a question</a>
                    <div>When you reach a blocking issue with your analysis using Hail, and you think you are unable to find an answer to your question via the documentation, search through or ask a question on our Forum! It is highly recommended -- your question may be able to serve another person in our ever growing Hail community.</div>
                </section>
            </div>
        </span>
    </xsl:template>
</xsl:stylesheet>
