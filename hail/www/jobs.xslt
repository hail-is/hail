<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:import href="template.xslt"/>

    <xsl:template name="page-title">Hail Jobs</xsl:template>
    <xsl:template name="meta-description"><meta name="description" content="Hail Jobs"/></xsl:template>
    <xsl:template name="navbar-script">
        <script>
            $(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li")
            .removeClass("active"); $("#jobs").addClass("active");});});
        </script>
    </xsl:template>

</xsl:stylesheet>