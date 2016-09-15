var fs = require('fs');

function parseOptionKey(str) {
	var p = str.split(' ').map(function(s){
		return s.replace(/\(/,'').replace(/\)/, '').replace(/^\-*/,'');
	}).reduce(function (a, b) {return a.length < b.length ? a : b; });
	return p;
}

function definitionListOption(attr, key) {
	var optType = attr.type.toLowerCase();

	if (attr.defaultValue == null || optType === "boolean") {defaultValue = "";}
	else if (optType === "string") {defaultValue = " (default: \"" + attr.defaultValue + "\")";}
	else {defaultValue = " (default: " + attr.defaultValue + ")";};

	var metaVar = attr.metaVar != null ? " " + attr.metaVar : "";

	return "<dt class=optkey>" + key + metaVar + "</dt>" + "<dd class=optdef>" + attr.usage + defaultValue + "</dd>";
}

function buildHeader(commandName, cmdId) {
	return "<a name=" + cmdId + "></a><h1 class=cmdhead>" + commandName + " <span style=\"font-size: 50%; vertical-align: middle; color: #555\">[ <a href=\"https://github.com/hail-is/hail/edit/master/docs/commands/"+cmdId+".md\" target=\"_blank\">edit</a> ]</span></h1>";
}

function buildDescription(cmdId, data) {
	var description = (data.description.endsWith('.')) ? data.description : data.description + ".";
	var multiAllelicString = (data.supportsMultiallelic) ? "Multi-allelic variants are supported." : "Multi-allelic variants are not supported. Use the <a href=#splitmulti>splitmulti</a> command before using this command.";
	return "<p class=description id=" + cmdId + ">" + description + " " + multiAllelicString + "</p>";
}

function buildCommandOptions(cmdId, options) {

	var optKeys = Object.keys(options);
	optKeys.sort(function(k1, k2){
	 	if (parseOptionKey(k1) < parseOptionKey(k2)) {return -1;}
	 	else if (parseOptionKey(k1) > parseOptionKey(k2)) {return 1}
	 	else {return 0};
	});

	var requiredKeys = optKeys.filter(function(k){return options[k].required;});
	var optionalKeys = optKeys.filter(function(k){return !options[k].required;});

	requiredKeyDefList = "<div class=opt_required><h4 class=required>Required</h4><dl class=options id=" + cmdId + ">" + requiredKeys.map(function(k) {return definitionListOption(options[k], k);}).join('') + "</dl></div>";
	optionalKeyDefList = "<div class=opt_optional><h4 class=optional>Optional</h4><dl class=options id=" + cmdId + ">" + optionalKeys.map(function(k) {return definitionListOption(options[k], k);}).join('') + "</dl></div>";

	return "<h3 class=opthead id=" + cmdId + ">Options:</h3><div class=optlistcont>" + requiredKeyDefList + optionalKeyDefList + "</div>";
}

function buildSynopsis(data) {
	return "<h3 class=synopsishead>Usage:</h3><p class=synopsis><pre class=synopsis><code>" + data.synopsis + "</code></pre></p>";
}

exports.buildGlobalOptions = function (options, $) {
      return $.map(options, definitionListOption);
}

exports.buildCommand = function (commandName, data, pandocOutputDir, $) {
    return new Promise(function (resolve, reject) {
        var cmdId = commandName.replace(/\s+/g, '_').replace(/\//, '_');
        var templateFile = pandocOutputDir + "commands/" + cmdId + ".html"

        function addContent() {
            $("div#" + cmdId + " div.cmdhead").append(buildHeader(commandName, cmdId));
            $("div#" + cmdId + " div.description").append(buildDescription(cmdId, data));
            $("div#" + cmdId + " div.options").append(buildCommandOptions(cmdId, data.options));
            $("div#" + cmdId + " div.synopsis").append(buildSynopsis(data));
        }

        $("body").append("<div class=command id=" + cmdId + "></div>");
        $("#toc-commands").append("<li><a href=#" + cmdId + ">" + commandName + "</a></li>");

        fs.exists(templateFile, function (exists) {
            if (!exists) {
                console.log("Warning: Did not find the file " + templateFile + "; Adding a generic template.");
                $("div#" + cmdId).append("<div class=cmdhead></div><div class=description></div><div class=synopsis></div><div class=options></div>");
                addContent();
                resolve();
            } else {
                $("div#" + cmdId).load(pandocOutputDir + "commands/" + cmdId + ".html", function (response, status, xhr) {
                    addContent();
                    resolve();
                });
            }
        });
    });

}
