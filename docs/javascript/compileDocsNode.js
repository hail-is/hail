#!/usr/bin/env node

'use strict';

process.on('uncaughtException', function (err) {
  console.log('Caught exception: ' + err);
  process.exit(1);
});

const docsHtmlTemplate = __dirname + "/" + process.argv[2];
const jsonCommandsFile = process.argv[3];
const pandocOutputDir = __dirname + "/" + process.argv[4];
const compiledHTMLOutputFile = __dirname + "/" + process.argv[5];

const jsdom = require('jsdom');
const fs = require('fs');

var $ = null;
var buildDocs = require("./buildDocs.js");
var mjAPI = require("mathjax-node/lib/mj-page.js");
var jsonData = require(jsonCommandsFile);

mjAPI.start();

jsdom.env(docsHtmlTemplate, function (err, window) {
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
                        resolve(response)
                    }
                    });
            });
        };

        function error(message) {
            console.log(message);
            process.exit(1);
        }

        var loadOverviewPromises = [loadReq("#Representation", pandocOutputDir + "Representation.html"),
                                    loadReq("#ImportingGenotypeData", pandocOutputDir + "Importing.html"),
                                    loadReq("#HailObjects", pandocOutputDir + "HailObjectProperties.html"),
                                    loadReq("#Annotations", pandocOutputDir + "Annotations.html"),
                                    loadReq("#HailExpressionLanguage", pandocOutputDir + "HailExpressionLanguage.html"),
                                    loadReq("#Filtering", pandocOutputDir + "Filtering.html"),
                                    loadReq("#ExportingData", pandocOutputDir + "ExportingData.html"),
                                    loadReq("#ExportingTSV", pandocOutputDir + "ExportTSV.html"),
                                    loadReq("#SQL", pandocOutputDir + "SQL.html"),
                                    loadReq("#GettingStarted", pandocOutputDir + "GettingStarted.html"),
                                    loadReq("#GlobalOptions", pandocOutputDir + "GlobalOptions.html")
                                    ];

        var buildCommandPromises = [];
        for (var commandName in jsonData.commands) {
            var data = jsonData.commands[commandName];
            if (!data.hidden) {
                buildCommandPromises.push(buildDocs.buildCommand(commandName, data, pandocOutputDir, $));
            }
        }

        Promise.all(loadOverviewPromises.concat(buildCommandPromises))
            .then(function() {$("div#GlobalOptions div.options").append(buildDocs.buildGlobalOptions(jsonData.global.options, $))}, error)
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