#!/usr/bin/env node

'use strict';

process.on('uncaughtException', function (err) {
  console.log('Caught exception: ' + err);
  process.exit(1);
});

const template = __dirname + "/" + process.argv[2];
const htmlInput = __dirname + "/" + process.argv[3];

const jsdom = require('jsdom');
const fs = require('fs');

buildIndex(template, "#body", htmlInput, __dirname + "/index.html");

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

function buildIndex(htmlTemplate, selector, pandocInput, outputFileName) {
    jsdom.env(htmlTemplate, function (err, window) {
        window.addEventListener("error", function (event) {
          console.error("script error!!", event.error);
          process.exit(1);
        });

        const $ = require('jquery')(window);

        var promise = loadReq(selector, pandocInput, $);

        promise.then(function() {
             $("h1#hail").remove();
             $('a[href*="badge"]').remove();
             $('a[href*="buildTypeStatusDiv"]').remove();
             $('a[href="https://hail.is"]').replaceWith("Hail");
             $("head").append('<script>$(document).ready(function () {$("#hail-navbar").load("navbar.html", function () {$(".nav li").removeClass("active"); $("#home").addClass("active");});});</script>');
             $('img[src="www/hail_spark_summit.png"]').replaceWith('<img src="hail_spark_summit.png"></img>');

             var html = $('html').html();
             fs.writeFile(outputFileName, html);

        }, error).catch(error);
    });
}
