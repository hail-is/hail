<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:import href="template.xslt"/>

    <xsl:template match="h1[@id='hail']">
    </xsl:template>

    <xsl:template match="a[contains(@href, 'badge')]">
    </xsl:template>

    <xsl:template match="a[contains(@href, 'buildTypeStatusDiv')]">
    </xsl:template>

    <xsl:template match="a[@href = 'https://hail.is']">
        Hail
    </xsl:template>

    <xsl:template name="page-title">Hail</xsl:template>
    <xsl:template name="meta-description">
        <meta name="description" content="Hail Overview"/>
    </xsl:template>
    <xsl:template name="navbar-script">
        <script>
            $(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li")
            .removeClass("active"); $("#home").addClass("active");});});
        </script>
    </xsl:template>
    <xsl:template name="jobs-banner">
        <a id="banner" href="jobs.html">
            <div id="banner-jobs" class="alert alert-info" role="alert">
                <p>Hail is hiring engineers!</p>
            </div>
        </a>
    </xsl:template>

</xsl:stylesheet>
