<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:template name="Duration" match="*/OverallResult|*/OverallResults">
        Duration: <xsl:value-of select="*/@durationInSeconds" />s
    </xsl:template>

    <xsl:template name="SourceInfo">
        <div class="source-info"><xsl:value-of select="@filename"/>:<xsl:value-of select="@line"/></div>
    </xsl:template>

    <xsl:template name="OverallResults" match="*/OverallResults">
            Successes: <xsl:value-of select="OverallResults/@successes" />
            Failures: <xsl:value-of select="OverallResults/@failures" />
            <xsl:if test="not(OverallResults/@expectedFailures=0)">Expected Failures: <xsl:value-of select="OverallResults/@expectedFailures" /></xsl:if>
    </xsl:template>

    <xsl:template name="test-result">
        <xsl:param name="success">true</xsl:param>
        <xsl:choose>
            <xsl:when test='$success="true"'>
                <div class="successful-test test-success-value">SUCCESS</div>
            </xsl:when>
            <xsl:otherwise>
                <div class="failed-test test-success-value">FAILED</div>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <xsl:template name="Info" match="Info" mode="testNode">
        <div class="info"><xsl:value-of select="."/></div>
    </xsl:template>

    <xsl:template name="Warning" match="Warning" mode="testNode">
        <div class="warning"><xsl:value-of select="."/></div>
    </xsl:template>

    <xsl:template name="Expression" match="Expression" mode="testNode">
        <div class="Expression result">
            <xsl:call-template name="test-result">
                <xsl:with-param name="success"><xsl:value-of select="@success"/></xsl:with-param>
            </xsl:call-template>
            <div class="header-text"><xsl:value-of select="@type"/>:
                <xsl:value-of select="Original"/>
                <i>(Expanded: <xsl:value-of select="Expanded"/>)
                </i>
            </div>
            <xsl:call-template name="SourceInfo"/>
            <br/>
        </div>
    </xsl:template>

    <xsl:template name="Section" match="Section" mode="testNode">
        <xsl:variable name="uid">
            <xsl:value-of select="generate-id(.)"/>
        </xsl:variable>
        <div class="result">
            <xsl:call-template name="test-result">
                <xsl:with-param name="success"><xsl:value-of select="string(OverallResults/@failures=OverallResults/@expectedFailures)"/></xsl:with-param>
            </xsl:call-template>
                <div class="header-text">Section:
                    <xsl:value-of select="@name"/>
                    <input type="button" value="v" onclick="javascript:unhide('{$uid}')" class="results-button"/>
                </div>
                <div class="section-results"><xsl:call-template name="OverallResults" /></div>
                <div class="duration"><xsl:call-template name="Duration"/></div>
                <xsl:call-template name="SourceInfo"/>
            </div>
            <div id="{$uid}" class="Section hidden">
                <xsl:apply-templates mode="testNode"/>
            </div>
    </xsl:template>

    <xsl:template name="TestCase" match="TestCase">
        <xsl:variable name="uid">
            <xsl:value-of select="generate-id(.)"/>
        </xsl:variable>
            <div class="result">
                <xsl:call-template name="test-result">
                    <xsl:with-param name="success"><xsl:value-of select="OverallResult/@success"/></xsl:with-param>
                </xsl:call-template>

                <div class="header-text">Test Case:
                    <xsl:value-of select="@name"/><xsl:value-of select="@description"/>
                    <input type="button" value="v" onclick="javascript:unhide('{$uid}')" class="results-button"/>
                </div>
                <div class="duration"><xsl:call-template name="Duration"/></div>
                <xsl:call-template name="SourceInfo"/>
            </div>
            <div id="{$uid}" class="Section hidden">
                <xsl:apply-templates mode="testNode"/>
            </div>

    </xsl:template>

    <xsl:key name="files" match="TestCase" use="@filename"/>

    <xsl:template match="/">
        <html>
            <head>
                <meta charset="utf-8"/>
                <title>C++ Catch2 Test Summary</title>
                <link href="style.css" rel="stylesheet" type="text/css"/>
                <script type="text/javascript">
                    function unhide(divID) {
                    var item = document.getElementById(divID);
                    if (item) {
                    item.classList.toggle('hidden');
                    }}
                </script>
            </head>
            <body>
                <h2>Tests</h2>
                <xsl:for-each select="Catch/Group">

                    <xsl:call-template name="OverallResults"/>

                    <xsl:for-each select="TestCase[count(. | key('files', @filename)[1]) = 1]">
                        <xsl:variable name="file" select="@filename"/>
                        <xsl:variable name="test-cases-in-file" select="key('files',$file)"/>
                        <h3><xsl:value-of select="@filename"/></h3>
                        <div class="file Section">
                        <xsl:for-each select="$test-cases-in-file">
                            <xsl:call-template name="TestCase"/>
                        </xsl:for-each>
                        </div>
                    </xsl:for-each>
                </xsl:for-each>
            </body>
        </html>

    </xsl:template>

</xsl:stylesheet>
