function searchTable(table_name, search_bar_name) {
  var searchTerms = document.getElementById(search_bar_name);
  var filter = searchTerms.value.toLowerCase();
  var table = document.getElementById(table_name);
  var tableBodies = table.tBodies;

  for (let tableBody of tableBodies) {
    for (let bodyRow of tableBody.rows) {
      let cells = bodyRow.getElementsByTagName("td");
      var anyMatch = false;
      for (let cell of cells) {
        if ((cell.textContent || cell.innerText).toLowerCase().indexOf(filter) >= 0) {
          anyMatch = true;
          break;
        }
      }
      if (anyMatch) {
        bodyRow.style.display = ""
      } else {
        bodyRow.style.display = "none"
      }
    }
  }
}
