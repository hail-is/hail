(function () {
    var container = document.getElementById('auto-refresh');
    if (!container) return;

    var detail = document.getElementById('auto-refresh-detail');
    var checkbox = document.getElementById('auto-refresh-toggle');
    var timer = null;

    function scheduleRefresh() {
        var target = new Date(Date.now() + 60000);
        detail.textContent = '(at ' + target.toLocaleTimeString() + ')';
        timer = setTimeout(function () {
            location.reload();
        }, 60000);
    }

    checkbox.addEventListener('change', function () {
        if (checkbox.checked) {
            scheduleRefresh();
        } else {
            clearTimeout(timer);
            timer = null;
            detail.textContent = '(every minute)';
        }
    });

    scheduleRefresh();
})();
