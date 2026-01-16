if (Object.hasOwn(window, "Plotly")) {
    if (document.getElementById('plotly-job-durations') != null) {
        var durationGraph = JSON.parse(document.getElementById('plotly-job-durations').dataset.graph);
        window.Plotly.plot('plotly-job-durations', durationGraph, {});
    }

    if (document.getElementById('plotly-job-resource-usage') != null) {
        var resourceUsageGraph = JSON.parse(document.getElementById('plotly-job-resource-usage').dataset.graph);
        window.Plotly.plot('plotly-job-resource-usage', resourceUsageGraph, {});
    }
}
