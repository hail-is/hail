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

const buildDocs = require("./buildDocs.js");
const mjAPI = require("mathjax-node/lib/mj-page.js");
const jsonData = require(jsonCommandsFile);

mjAPI.start();

jsdom.env(docsHtmlTemplate, function (err, window) {
        window.addEventListener("error", function (event) {
          console.error("script error!!", event.error);
          process.exit(1);
        });

        const $ = require('jquery')(window);

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

        const loadOverviewPromises = ["Representation",
                                    "Importing",
                                    "HailObjectProperties",
                                    "Annotations",
                                    "HailExpressionLanguage",
                                    "Filtering",
                                    "ExportingData",
                                    "ExportingTSV",
                                    "SQL",
                                    "GettingStarted",
                                    "GlobalOptions"]
                                    .map(name => loadReq("#" + name, pandocOutputDir + name + ".html"));

        const buildCommandPromises = jsonData.commands
            .filter(command => !command.hidden)
            .map(command => buildDocs.buildCommand(command, pandocOutputDir, $));


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
                    const HTML = "<!DOCTYPE html>\n" + document.documentElement.outerHTML.replace(/^(\n|\s)*/, "");

                    fs.writeFile(compiledHTMLOutputFile, HTML);
                 });
            }, error)
            .catch(error);
});