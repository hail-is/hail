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
                    <h1 id="logo-title" style='background: inherit; color: inherit;animation: none'>Cheatsheets</h1>
                    <subtitle id="logo-subtitle" style='background:inherit;animation: none; opacity: 1'>Shortcuts to plinking through Hail</subtitle>
                </div>
            </div>
            <div id="about" class="about">
                <p>
                Hail has two cheatsheets, describing the two data structures in Hail: the Table and the MatrixTable.
                </p>
                <section class='left'>
                    <a class='button' href='/docs/0.2/_static/cheatsheets/hail_tables_cheat_sheet.pdf'>Tables</a>
                    <div>Tables are the Hail data structure for one-dimensional data. You can create a Table from TSVs, CSVs, sites VCFs, FAM files, and Pandas DataFrames.</div>
                </section>
                <section class='left'>
                    <a class='button' href='/docs/0.2/_static/cheatsheets/hail_matrix_tables_cheat_sheet.pdf'>MatrixTables</a>
                    <div>MatrixTables are the Hail data structure for two-dimensional data. You can create a MatrixTable from VCF, BGEN, and PLINK files.</div>
                </section>
            </div>
        </span>
    </xsl:template>
</xsl:stylesheet>
