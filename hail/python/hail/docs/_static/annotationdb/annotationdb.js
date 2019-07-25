$("#checkAll").click(function(){
    $('input:checkbox').not(this).prop('checked', this.checked);
});

$.ajax({
        type: 'GET',
        url: 'https://www.googleapis.com/storage/v1/b/hail-common/o/annotationdb%2f1%2fannotation_db.json?alt=media',
        dataType: 'json',
        success: function (data) {
             console.log(data);
             var tr = data.report
             var dict = {};
             var mySet = new Set();
             for (var i = 0; i < data.length; i++) {
                 if(data[i].name in dict){
                    dict[data[i].name].push(data[i].reference_genome);
                 }else{
                     dict[data[i].name] = [data[i].reference_genome]
                 }  
            }
            for (var i = 0; i < data.length; i++) {
                    if(!mySet.has(data[i].name)){
                        mySet.add(data[i].name);
                        tr = $('<tr/>');
                        tr.append("<td><input type='checkbox' class='checkboxadd' value='"+data[i].name+"' onClick='updateTextArea()'/>&nbsp;</td>")
                        tr.append("<td>" + data[i].name + "</td>");
                        tr.append("<td>" + data[i].description + "\n<a href='"+ data[i].url + "'>" +data[i].url+ "</a></td>");
                        tr.append("<td>" + (data[i].version == null ? "" : data[i].version) + 
                                  "\n" + (dict[data[i].name][0] == null ? "" : "("+ dict[data[i].name]+ ")") + "</td>");
                        $('.table1').append(tr);
                     }
                }
        }
    });


function filterTable() {
    var input, filter, found, table, tr, td, i, j;
    input = document.getElementById("searchInput");
    filter = input.value.toUpperCase();
    table = document.getElementById("table1");
    tr = table.getElementsByTagName("tr");
    for (i = 0; i < tr.length; i++) {
        td = tr[i].getElementsByTagName("td");
        for (j = 0; j < td.length; j++) {
            if (td[j].innerHTML.toUpperCase().indexOf(filter) > -1) {
                found = true;
            }
        }
        if (found) {
            tr[i].style.display = "";
            found = false;
        } else if (!tr[i].id.match('^tableHeader'))  {
            tr[i].style.display = "none";
        }
    }
}

function copy() {
  let textarea = document.getElementById("result");
  textarea.select();
  document.execCommand("copy");
}


function updateTextArea() {
    var text = "db = hl.experimental.DB()\nmt = db.annotate_rows_db(mt,";
    var isFirst = true;
    $('input[type=checkbox]:checked').filter(".checkboxadd").each( function() {
        if(!isFirst){
            text += ",";
        } else {
            isFirst = false;
        } 
        text += '"' + $(this).val() + '"';
        $('#result').val(text + ')');
    });
     $('input[type="checkbox"]:not(:checked)').filter(".checkboxadd").each( function(){
        $('#result').val(text + ')');
    });
}

$('input[type=checkbox]').change(function () {
    updateTextArea();
}); 
