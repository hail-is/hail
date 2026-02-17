(function () {
    var container = document.getElementById('auto-refresh');
    if (!container) return;

    var timeSpan = document.getElementById('auto-refresh-time');
    var checkbox = document.getElementById('auto-refresh-toggle');
    var timer = null;

    function scheduleRefresh() {
        var target = new Date(Date.now() + 60000);
        timeSpan.textContent = target.toLocaleTimeString();
        timeSpan.style.display = '';
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
            timeSpan.textContent = '';
            timeSpan.style.display = 'none';
        }
    });

    scheduleRefresh();
})();
