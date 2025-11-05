if (document.getElementById('plotly-timeseries') !== undefined) {
    var graphJson = JSON.parse(document.getElementById('plotly-timeseries').dataset.graph);
    Plotly.plot('plotly-timeseries', graphJson, {});
}