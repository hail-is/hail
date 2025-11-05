if (document.getElementById('plotly-job-durations') !== undefined) {
    var durationGraph = JSON.parse(document.getElementById('plotly-job-durations').dataset.graph);
    Plotly.plot('plotly-job-durations', durationGraph, {});
}

if (document.getElementById('plotly-job-resource-usage') !== undefined) {
    var resourceUsageGraph = JSON.parse(document.getElementById('plotly-job-resource-usage').dataset.graph);
    Plotly.plot('plotly-job-resource-usage', resourceUsageGraph, {});
}
