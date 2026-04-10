(function () {
    var enableReactBtn = document.getElementById('enable-react-ui-btn');
    if (enableReactBtn) {
        enableReactBtn.addEventListener('click', function (e) {
            e.preventDefault();
            document.cookie = 'hail_react_ui=1; max-age=108000; path=/; SameSite=Lax';
            location.reload();
        });
    }
})();

if (Object.hasOwn(window, "Plotly")) {
    if (document.getElementById('plotly-job-durations') != null) {
        var durationGraph = JSON.parse(document.getElementById('plotly-job-durations').dataset.graph);
        window.Plotly.newPlot('plotly-job-durations', durationGraph, {});
    }

    if (document.getElementById('plotly-job-resource-usage') != null) {
        var resourceUsageGraph = JSON.parse(document.getElementById('plotly-job-resource-usage').dataset.graph);
        window.Plotly.newPlot('plotly-job-resource-usage', resourceUsageGraph, {});
    }
}
