<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="html" encoding="utf-8" indent="yes" />

<xsl:template match="h1[@id='hail']">
</xsl:template>

<xsl:template match="a[contains(@href, 'badge')]">
</xsl:template>

<xsl:template match="a[contains(@href, 'buildTypeStatusDiv')]">
</xsl:template>

<xsl:template match="a[@href = 'https://hail.is']">
Hail
</xsl:template>

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
      <meta charset="utf-8" />
      <title>Hail</title>
      <link rel='shortcut icon' href='hail_logo_sq.png' type='image/x-icon' />
      <meta name="description" content="Hail Overview" />
      <script  src="jquery-3.1.1.min.js"></script>
      <script src="bootstrap.min.js"></script>
      <link rel="stylesheet" href="bootstrap.min.css" type="text/css" />
      <link rel="stylesheet" href="style.css" />
      <link rel="stylesheet" href="navbar.css" />
      <script>
        $(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li").removeClass("active"); $("#home").addClass("active");});});
      </script>
    </head>

    <body>
      <nav class="navbar navbar-default navbar-static-top" id="hail-navbar"></nav>
      <div id="body">
        <a id="banner" href="jobs.html">
          <div id="banner-jobs" class="alert alert-info" role="alert">
            <p>Hail is hiring!</p>
          </div>
        </a>
        <xsl:apply-templates select="body"/>
      </div>
    </body>
  </html>
</xsl:template>
</xsl:stylesheet>
