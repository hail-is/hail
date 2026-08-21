(function () {
  var enableBtn = document.getElementById('enable-react-ui-btn');
  if (enableBtn) {
    enableBtn.addEventListener('click', function (e) {
      e.preventDefault();
      document.cookie = 'hail_react_ui=1; max-age=2147483647; path=/; SameSite=Lax';
      location.reload();
    });
  }
  var disableBtn = document.getElementById('disable-react-ui-btn');
  if (disableBtn) {
    disableBtn.addEventListener('click', function (e) {
      e.preventDefault();
      document.cookie = 'hail_react_ui=; max-age=0; path=/; SameSite=Lax';
      location.reload();
    });
  }
}());
