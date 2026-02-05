if (Object.hasOwn(window, "Plotly")) {
    if (document.getElementById('plotly-timeseries') !== undefined) {
        var graphJson = JSON.parse(document.getElementById('plotly-timeseries').dataset.graph);
        window.Plotly.plot('plotly-timeseries', graphJson, {});
    }
}
