function searchTable(table_name, search_bar_name) {
  var searchTerms = document.getElementById(search_bar_name);
  var filter = searchTerms.value.toLowerCase();
  var table = document.getElementById(table_name);
  var tableRecords = table.getElementsByTagName("tr");

  for (var i = 1; i < tableRecords.length; ++i) {
    var record = tableRecords[i];
    var tds = record.getElementsByTagName("td");
    var anyMatch = false;
    for (var j = 0; j < tds.length; ++j) {
      var td = tds[j]
      if ((td.textContent || td.innerText).toLowerCase().indexOf(filter) >= 0) {
        anyMatch = true;
        break;
      }
    }
    if (anyMatch) {
      if (record.parentNode.style.display == "none") {
        wrapper = record.parentNode

        row_container = wrapper.parentNode
        row_container.insertBefore(record, wrapper)
        row_container.removeChild(wrapper)
      }
    } else {
      if (record.parentNode.style.display != "none") {
        wrapper = document.createElement('div')
        wrapper.style.display = "none"

        row_container = record.parentNode
        row_container.insertBefore(wrapper, record)
        row_container.removeChild(record)

        wrapper.appendChild(record)
      }
    }
  }
}
