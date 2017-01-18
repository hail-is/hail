var fs = require('fs');

exports.buildFaqHeader = function (name, $) {
    return new Promise(function (resolve, reject) {
        $("#" + name + " h2").append("<span style=\"font-size: 50%; vertical-align: middle; color: #555\"> [ <a href=\"https://github.com/hail-is/hail/edit/master/docs/faq/"+name+".md\" target=\"_blank\">edit</a> ]</span>");
        resolve();
    });
}

exports.buildFaqTOC = function ($) {
  return new Promise(function (resolve, reject) {
      function listItem(id, text) { return "<li><a href=#" + id + ">" + text + "</a></li>"; }
      function anchor(id) { return "<a name=" + id + "></a>"; }

      $("h4").each(function () {
          var element = $(this);
          var id = element.attr("id");
          var text = element.text();
          $("#TOC ul").append(listItem(id, text));
          element.prepend(anchor(id));
          element.addClass("faq");
      });

      resolve();
  });
}
