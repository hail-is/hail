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
                <link rel='shortcut icon' href='/hail_logo_sq-sm-opt.ico' type='image/x-icon'/>
                <xsl:call-template name="meta-description"/>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/docsearch.js@2/dist/cdn/docsearch.min.css" />
                <link rel="stylesheet" href="/style.css"/>
                <link rel="stylesheet" href="/navbar.css"/>
                <xsl:call-template name="header"/>
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
                                <img alt="Hail" height="30" id="logo" src="/hail-logo-cropped-sm-opt.png" />
                            </a>
                            <button type="button" id='navbar-toggler' class="navbar-toggler" data-toggle="collapse" data-target="#hail-navbar-collapse" aria-expanded="false">
                                <span class="icon-bar"></span>
                                <span class="icon-bar"></span>
                                <span class="icon-bar"></span>
                            </button>
                        </div>
                        <div class="collapse navbar-collapse" id="hail-navbar-collapse">
                            <input id='search' type='search' placeholder='Search Hail Docs'/>
                            <ul class="nav navbar-nav navbar-right" id="hail-menu">
                                <li class="nav-item">
                                    <a href="/docs/0.2/index.html">Docs</a>
                                </li>
                                <li class="nav-item">
                                    <a href="https://discuss.hail.is">Forum</a>
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
                            <xsl:text disable-output-escaping="yes">
                                <![CDATA[
                                    const cached = document.getElementById("hail-navbar-collapse");
                                    const initialStyle = cached.style.display;
                                    document.getElementById("navbar-toggler").addEventListener("click", () => {
                                        const computed = getComputedStyle(cached);

                                        if (computed.display == 'none') {
                                            cached.style.display = 'block';
                                        } else {
                                            cached.style.display = initialStyle;
                                        }
                                    });
                                    (function () {
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
                        <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/docsearch.js@2/dist/cdn/docsearch.min.js" async="true"></script>
                        <script type="text/javascript">
                            <xsl:text disable-output-escaping="yes" >
                                <![CDATA[
                                    let isHighlighted = false;
                                    const cachedSearchInput = document.getElementById("search");

                                    const cachedNavbar = document.getElementById("hail-navbar")
                                    cachedSearchInput.addEventListener("keyup", (ev) => {
                                        handleSearchKeyUp(cachedSearchInput.value, ev)
                                    });

                                    function handleSearchKeyUp(query, ev) {
                                        if(ev.keyCode == 13 && !isHighlighted) {
                                            location.href = `/search.html?query=${query}`;
                                        }
                                    }

                                    window.addEventListener("keyup", (ev) => {
                                        if(ev.keyCode != 191) {
                                            return;
                                        }

                                        cachedSearchInput.focus();
                                    })

                                    const algoliaOptions = {
                                        hitsPerPage: 10,
                                        exactOnSingleWordQuery: "word",
                                        queryType: "prefixAll",
                                        advancedSyntax: true,
                                    };

                                    function run() {
                                        docsearch({
                                            apiKey: 'd2dee24912091336c40033044c9bac58',
                                            indexName: 'hail_is',
                                            inputSelector: '#search',
                                            debug: false, // hide on blur
                                            handleSelected: function(input, event, suggestion, datasetNumber, context) {
                                                isHighlighted = !!suggestion;
                                                location.href = suggestion.url;
                                            },
                                            queryHook: function(query) {
                                                // algolia seems to split on period, but not split queries on period, affects methods search
                                                return query.replace(/\./g, " ");
                                            },
                                            autocompleteOptions: {
                                                autoselect: false
                                            },
                                            algoliaOptions: algoliaOptions,
                                        });

                                        const cachedAlgolia = document.querySelector("#algolia-autocomplete-listbox-0 > .ds-dataset-1");

                                        cachedAlgolia.style.overflow = 'scroll';

                                        cachedAlgolia.style.maxHeight = `${window.innerHeight - cachedNavbar.offsetHeight}px`;
                                        let evTimeout = null;
                                        const ev = window.addEventListener("resize", () => {
                                            if (evTimeout) {
                                                clearTimeout(evTimeout);
                                            }

                                            evTimeout = setTimeout(() => {
                                                cachedAlgolia.style.maxHeight = `${window.innerHeight - cachedNavbar.offsetHeight}px`;
                                                evTimeout = null;
                                            }, 100);
                                        })
                                    }

                                    function check()  {
                                        if(window.docsearch)  {
                                            run();
                                            return;
                                        }

                                        setTimeout(check, 16);
                                    }

                                    check();
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
