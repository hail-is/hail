(function () {
  var enableBtn = document.getElementById('enable-react-ui-btn');
  if (enableBtn) {
    enableBtn.addEventListener('click', function (e) {
      e.preventDefault();
      document.cookie = 'hail_react_ui_trial=1; max-age=2147483647; ' + enableBtn.dataset.cookieAttrs;
      location.reload();
    });
  }
}());
