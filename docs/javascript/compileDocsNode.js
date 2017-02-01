#!/usr/bin/env node

'use strict';

process.on('uncaughtException', function (err) {
  console.log('Caught exception: ' + err);
  process.exit(1);
});

const faqHtmlTemplate = __dirname + "/" + process.argv[2];
const template = __dirname + "/" + process.argv[3];
const pandocOutputDir = __dirname + "/" + process.argv[4];

const jsdom = require('jsdom');
const fs = require('fs');

const buildDocs = require("./buildDocs.js");
const mjAPI = require("mathjax-node/lib/mj-page.js");

mjAPI.start();

buildFAQ(faqHtmlTemplate, __dirname + "/faq.html");

buildSinglePage(template, "#body", pandocOutputDir + "tutorial/Tutorial.html",  __dirname + "/tutorial.html",
    '<script>$(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li").removeClass("active"); $("#docs").addClass("active"); $("#tutorial").addClass("active");});});</script>');

buildSinglePage(template, "#body", pandocOutputDir + "reference/HailExpressionLanguage.html",  __dirname + "/expr_lang.html",
    '<script>$(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li").removeClass("active"); $("#docs").addClass("active"); $("#exprlang").addClass("active");});});</script>');

buildIndex(template, "#body", "README.html", __dirname + "/index.html");


function error(message) {
    console.log(message);
    process.exit(1);
}

function loadReq(selector, file, $) {
    return new Promise(function (resolve, reject) {
        const s = $(selector);
        if (s.length == 0) {
            reject("No elements found for selector " + selector);
        } else {
            $(selector).load(file, function (response, status, xhr) {
                if (status == "error") {
                    console.log("error when loading file: " + file);
                    reject(status)
                } else {
                    resolve(response)
                }
            });
        }
    });
};

function runMathJax(document, callback) {
    mjAPI.typeset({
        html: document.body.innerHTML,
        renderer: "SVG",
        inputs: ["TeX"],
        xmlns: "mml"
     }, function(result) {
        document.body.innerHTML = result.html;
        const HTML = "<!DOCTYPE html>\n" + document.documentElement.outerHTML.replace(/^(\n|\s)*/, "");
        callback(HTML);
    })
}

function buildFAQ(htmlTemplate, outputFileName) {
    jsdom.env(htmlTemplate, function (err, window) {
        window.addEventListener("error", function (event) {
          console.error("script error!!", event.error);
          process.exit(1);
        });

        const $ = require('jquery')(window);

    var faqNames = ["Annotations",
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
                    "OptimizePipeline"];

    var loadFaqPromises = faqNames.map(name => loadReq("#" + name, pandocOutputDir + "faq/" + name + ".html", $));

        Promise.all(loadFaqPromises)
            .then(function () {
                buildDocs.buildFaqTOC($);
                faqNames.map(name => buildDocs.buildFaqHeader(name, $));
            }, error)
            .then(function() {
                 var document = jsdom.jsdom($('html').html());
                 runMathJax(document, function(html) {
                    fs.writeFile(outputFileName, html);
                 });
            }, error)
            .catch(error);
    });
}

function buildSinglePage(htmlTemplate, selector, pandocInput, outputFileName, scriptTag) {
    jsdom.env(htmlTemplate, function (err, window) {
        window.addEventListener("error", function (event) {
          console.error("script error!!", event.error);
          process.exit(1);
        });

        const $ = require('jquery')(window);

        var loadTutorialPromises = [loadReq(selector, pandocInput, $)];

        Promise.all(loadTutorialPromises)
            .then(function() {
                $("head").append(scriptTag);
                 var document = jsdom.jsdom($('html').html());
                 runMathJax(document, function(html) {
                    fs.writeFile(outputFileName, html);
                 });
            }, error)
            .catch(error);
    });
}

function buildIndex(htmlTemplate, selector, pandocInput, outputFileName) {
    jsdom.env(htmlTemplate, function (err, window) {
        window.addEventListener("error", function (event) {
          console.error("script error!!", event.error);
          process.exit(1);
        });

        const $ = require('jquery')(window);

        var loadTutorialPromises = [loadReq(selector, pandocInput, $)];

        Promise.all(loadTutorialPromises)
            .then(function() {
                 $("h1#hail").remove();
                 $('a[href*="badge"]').remove();
                 $('a[href*="buildTypeStatusDiv"]').remove();
                 $('a[href="https://hail.is"]').replaceWith("Hail");
                 $("head").append('<script>$(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li").removeClass("active"); $("#home").addClass("active");});});</script>')

                 var document = jsdom.jsdom($('html').html());

                 runMathJax(document, function(html) {
                    fs.writeFile(outputFileName, html);
                 });
            }, error)
            .catch(error);
    });
}
