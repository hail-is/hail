if (Object.hasOwn(window, "Plotly")) {
    const durationEl = document.getElementById('plotly-job-durations');
    if (durationEl != null) {
        const graph = JSON.parse(durationEl.dataset.graph);
        window.Plotly.newPlot('plotly-job-durations', graph.data, graph.layout, {responsive: true});
    }

    const resourceChartIds = [
        'plotly-resource-cpu',
        'plotly-resource-memory',
        'plotly-resource-net-down',
        'plotly-resource-net-up',
        'plotly-resource-storage-overlay',
        'plotly-resource-storage-io',
    ];
    for (const id of resourceChartIds) {
        const el = document.getElementById(id);
        if (el != null) {
            const graph = JSON.parse(el.dataset.graph);
            window.Plotly.newPlot(id, graph.data, graph.layout, {responsive: true});
        }
    }
}
