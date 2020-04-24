<?xml version="1.0" encoding="ISO-8859-15"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" encoding="utf-8" indent="yes"/>
    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>
    <xsl:template match="html">
        <xsl:text disable-output-escaping='yes'>&lt;!DOCTYPE html&gt;</xsl:text>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <title><xsl:call-template name="page-title"/></title>
                <link rel='shortcut icon' href='hail_logo_sq.ico' type='image/x-icon'/>
                <xsl:call-template name="meta-description"/>
                <link rel="stylesheet" href="/style.css"/>
                <link rel="stylesheet" href="/navbar.css"/>
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
                <nav class="navbar align-content-start justify-content-start sticky" id="hail-navbar">
                    <div class="container-fluid align-content-start justify-content-start d-flex" id="hail-container-fluid">
                        <div class="navbar-header" id="hail-navbar-header">
                            <a class="navbar-left" id="hail-navbar-brand" href="/">
                                <img alt="Hail" height="30" id="logo" src="/hail-logo-cropped.png" />
                            </a>
                            <button type="button" id='navbar-toggler' class="navbar-toggler" data-toggle="collapse" data-target="#hail-navbar-collapse" aria-expanded="false">
                                <span class="icon-bar"></span>
                                <span class="icon-bar"></span>
                                <span class="icon-bar"></span>
                            </button>
                        </div>
                        <div class="collapse navbar-collapse" id="hail-navbar-collapse">
                            <ul class="nav navbar-nav navbar-right" id="hail-menu">
                                <li class="nav-item">
                                    <a href="/docs/0.2/index.html">Docs</a>
                                </li>
                                <li class="nav-item">
                                    <a href="https://discuss.hail.is">Forum</a>
                                </li>
                                <li class="nav-item">
                                    <a href="https://hail.zulipchat.com">Chat</a>
                                </li>
                                <li class="nav-item">
                                    <a href="/references.html">Powered-Science</a>
                                </li>
                                <li class="nav-item">
                                    <a href="https://blog.hail.is/">Blog</a>
                                </li>
                                <li class="nav-item">
                                    <a href="https://workshop.hail.is">Workshop</a>
                                </li>
                                <li class="nav-item" style='margin-top:-2px'>
                                    <a href="https://github.com/hail-is/hail" class='img-link' target='_blank'>
                                        <img width="20" src='/GitHub-Mark-64px.png'/>
                                    </a>
                                </li>
                            </ul>
                        </div>
                        <script>
                            <xsl:text disable-output-escaping="yes" >
                                <![CDATA[
                                    const cached = document.getElementById("hail-navbar-collapse");
                                    document.getElementById("navbar-toggler").addEventListener("click", () => {
                                        const computed = getComputedStyle(cached);

                                        if (computed.display == 'none') {
                                        cached.style.display = 'block';
                                        } else {
                                        cached.style.display = 'none';
                                        }
                                    });
                                    (function(){
                                        var cpage = location.pathname;
                                        var menuItems = document.querySelectorAll('#hail-menu a');

                                        for (var i = 0; i < menuItems.length; i++) {
                                            if (menuItems[i].pathname === cpage && menuItems[i].host == location.host) {
                                                menuItems[i].className = "active";
                                                return;
                                            }
                                        }

                                        if (cpage === "/" || cpage === "/index.html") {
                                            document.getElementById('hail-navbar-brand').className = "active";
                                        };
                                    })();
                                ]]>
                            </xsl:text>
                        </script>
                    </div>
                </nav>
                <xsl:apply-templates select="body"/>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>
