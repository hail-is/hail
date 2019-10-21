$("#checkAll").click(function(){
    $('input:checkbox').not(this).prop('checked', this.checked);
});

$.ajax({type: 'GET',
        url: ('https://www.googleapis.com/storage/v1/b/hail-common/o/annotationdb%2f' +
              hail_version +
              '%2fannotation_db.json?alt=media'),
        dataType: 'json',
        success: function (data) {
          for (name in data) {
            dataset = data[name]
            versions_string = dataset.versions.map(function (i) {
              return i["version"]
            }).reduce(function (i, j) {
              return i + "; " + j
            })
            tr = $('<tr/>');
            tr.append("<td><input type='checkbox' class='checkboxadd' value='"+name+"' onClick='updateTextArea()'/>&nbsp;</td>")
            tr.append("<td>" + name + "</td>");
            tr.append("<td>" + dataset.description + "\n<a href='"+ dataset.url + "'>link</a></td>");
            tr.append("<td>" + versions_string + "</td>");
            $('.table1').append(tr);
          }
        }
       });


function filterTable() {
  let input = document.getElementById("searchInput")
  let filter = input.value.toUpperCase()
  let table = document.getElementById("table1")
  let tr = table.getElementsByTagName("tr")
  var found = false
  for (var i = 0; i < tr.length; i++) {
    let td = tr[i].getElementsByTagName("td")
    for (var j = 0; j < td.length; j++) {
      if (td[j].innerHTML.toUpperCase().indexOf(filter) > -1) {
        found = true
      }
    }
    if (found) {
      tr[i].style.display = ""
      found = false
    } else if (!tr[i].id.match('^tableHeader'))  {
      tr[i].style.display = "none"
    }
  }
}

function copy() {
  let textarea = document.getElementById("result");
  textarea.select();
  document.execCommand("copy");
}


function updateTextArea() {
    var text = "db = hl.experimental.DB()\nmt = db.annotate_rows_db(mt";
    $('input[type=checkbox]:checked').filter(".checkboxadd").each( function() {
        text += ', "' + $(this).val() + '"';
        $('#result').val(text + ')');
    });
     $('input[type="checkbox"]:not(:checked)').filter(".checkboxadd").each( function(){
        $('#result').val(text + ')');
    });
}

$('input[type=checkbox]').change(function () {
    updateTextArea();
});
