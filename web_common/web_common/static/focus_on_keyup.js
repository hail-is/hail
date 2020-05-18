function focusOnSlash(name) {
  document.body.onkeyup = function(e){
    if (e.keyCode == 191){
      document.getElementById(name).focus();
    }
  }
}
