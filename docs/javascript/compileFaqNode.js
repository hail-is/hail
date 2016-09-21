#!/usr/bin/env node

'use strict';

process.on('uncaughtException', function (err) {
  console.log('Caught exception: ' + err);
  process.exit(1);
});

const faqHtmlTemplate = __dirname + "/" + process.argv[2];
const pandocOutputDir = __dirname + "/" + process.argv[3];
const compiledHTMLOutputFile = __dirname + "/" + process.argv[4];

const jsdom = require('jsdom');
const fs = require('fs');

var $ = null;
var mjAPI = require("mathjax-node/lib/mj-page.js");

mjAPI.start();

jsdom.env(faqHtmlTemplate, function (err, window) {
        window.addEventListener("error", function (event) {
          console.error("script error!!", event.error);
          process.exit(1);
        });

        $ = require('jquery')(window);

        function loadReq(selector, file) {
            return new Promise(function (resolve, reject) {
                $(selector).load(file, function (response, status, xhr) {
                    if (status == "error") {
                        console.log("error when loading file: " + file);
                        reject(status)
                    } else {
                        resolve(response);
                    }
                });
            });
        }

        function error(message) {
            console.log(message);
            process.exit(1);
        }

        function buildTOC() {
            return new Promise(function (resolve, reject) {

                function listItem(id, text) { return "<li><a href=#" + id + ">" + text + "</a></li>"; }

                function anchor(id) { return "<a name=" + id + "></a>"; }

                $("h4").each(function () {
                    var element = $(this);
                    var id = element.attr("id");
                    var text = element.text();
                    $("#TOC ul").append(listItem(id, text));
                    element.prepend(anchor(id));
                });

                resolve();
            });
        }

        var loadFaqPromises =  ["Annotations",
                                "DataRepresentation",
                                "DevTools",
                                "ErrorMessages",
                                "ExportingData",
                                "ExpressionLanguage",
                                "FilteringData",
                                "General",
                                "ImportingData",
                                "Installation",
                                "Methods",
                                "OptimizePipeline",
                                "Representation"]
                                .map(name => loadReq("#" + name, pandocOutputDir + name + ".html"));;

        Promise.all(loadFaqPromises)
            .then(buildTOC, error)
            .then(function() {
                 var document = jsdom.jsdom($('html').html());

                 mjAPI.typeset({
                   html: document.body.innerHTML,
                   renderer: "SVG",
                   inputs: ["TeX"],
                   xmlns: "mml"
                 }, function(result) {
                   document.body.innerHTML = result.html;
                   var HTML = "<!DOCTYPE html>\n" + document.documentElement.outerHTML.replace(/^(\n|\s)*/, "");
                   fs.writeFile(compiledHTMLOutputFile, HTML);
                 });
            }, error).catch(error);
});