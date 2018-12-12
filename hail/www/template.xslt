<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" encoding="utf-8" indent="yes"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>


    <xsl:template match="html">
<xsl:text disable-output-escaping='yes'>&lt;!DOCTYPE html&gt;
</xsl:text>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <title><xsl:call-template name="page-title"/></title>
                <link rel='shortcut icon' href='hail_logo_sq.png' type='image/x-icon'/>
                <xsl:call-template name="meta-description"/>
                <script src="jquery-3.1.1.min.js"></script>
                <script src="bootstrap.min.js"></script>
                <link rel="stylesheet" href="bootstrap.min.css" type="text/css"/>
                <link rel="stylesheet" href="style.css"/>
                <link rel="stylesheet" href="navbar.css"/>
                <xsl:call-template name="navbar-script"/>
                <script>
                    (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
                    (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
                    m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
                    })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

                    ga('create', 'UA-86050742-1', 'auto');
                    ga('send', 'pageview');
                </script>
            </head>

            <body>
                <nav class="navbar navbar-default navbar-static-top" id="hail-navbar"></nav>
                <div id="body">
                    <xsl:call-template name="jobs-banner"/>
                    <xsl:apply-templates select="body"/>
                </div>
            </body>
        </html>
    </xsl:template>

    <xsl:template name="jobs-banner"/>

</xsl:stylesheet>
