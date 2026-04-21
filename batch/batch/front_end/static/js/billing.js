(function () {
    var exportBtn = document.getElementById('billing-export-btn');
    if (!exportBtn) return;

    var statusEl = document.getElementById('billing-export-status');
    var basePath = (document.getElementById('billing-js') || {}).dataset?.basePath ?? '';

    function toIso(mmddyyyy) {
        if (!mmddyyyy) {
            var now = new Date();
            return now.getFullYear() + '-'
                + String(now.getMonth() + 1).padStart(2, '0') + '-'
                + String(now.getDate()).padStart(2, '0');
        }
        var parts = mmddyyyy.split('/');
        if (parts.length !== 3) return mmddyyyy;
        return parts[2] + '-' + parts[0] + '-' + parts[1];
    }

    function csvEscape(val) {
        var s = String(val);
        if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\n') >= 0) {
            return '"' + s.replace(/"/g, '""') + '"';
        }
        return s;
    }

    function toCsv(rows, columns) {
        var lines = [columns.join(',')];
        for (var i = 0; i < rows.length; i++) {
            lines.push(columns.map(function (col) { return csvEscape(rows[i][col]); }).join(','));
        }
        return lines.join('\n');
    }

    function groupByProject(records) {
        var acc = {};
        records.forEach(function (r) {
            acc[r.billing_project] = (acc[r.billing_project] || 0) + r.total_spent;
        });
        return Object.keys(acc).sort().map(function (bp) {
            return { billing_project: bp, total_spent: acc[bp] };
        });
    }

    function groupByUser(records) {
        var acc = {};
        records.forEach(function (r) {
            acc[r.user] = (acc[r.user] || 0) + r.total_spent;
        });
        return Object.keys(acc).sort().map(function (u) {
            return { user: u, total_spent: acc[u] };
        });
    }

    function groupByProjectUser(records) {
        return records.slice().sort(function (a, b) {
            if (a.billing_project !== b.billing_project) {
                return a.billing_project < b.billing_project ? -1 : 1;
            }
            return a.user < b.user ? -1 : 1;
        });
    }

    function getGrouping() {
        var radios = document.querySelectorAll('input[name="export-grouping"]');
        for (var i = 0; i < radios.length; i++) {
            if (radios[i].checked) return radios[i].value;
        }
        return 'by-project-user';
    }

    function triggerDownload(csvText, filename) {
        var blob = new Blob([csvText], { type: 'text/csv' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    exportBtn.addEventListener('click', function () {
        exportBtn.disabled = true;
        statusEl.textContent = 'Fetching\u2026';
        statusEl.classList.remove('hidden');

        var query = window.location.search;
        fetch(basePath + '/api/v1alpha/billing' + query)
            .then(function (resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
            .then(function (records) {
                var grouping = getGrouping();
                var rows, columns, label;
                if (grouping === 'by-project') {
                    rows = groupByProject(records);
                    columns = ['billing_project', 'total_spent'];
                    label = 'by billing project';
                } else if (grouping === 'by-user') {
                    rows = groupByUser(records);
                    columns = ['user', 'total_spent'];
                    label = 'by user';
                } else {
                    rows = groupByProjectUser(records);
                    columns = ['billing_project', 'user', 'total_spent'];
                    label = 'by billing project and user';
                }

                var params = new URLSearchParams(window.location.search);
                var startStr = toIso(params.get('start') || '');
                var endStr = toIso(params.get('end') || '');
                var filename = 'Hail billing export ' + startStr + ' to ' + endStr + ' ' + label + '.csv';

                triggerDownload(toCsv(rows, columns), filename);
                statusEl.classList.add('hidden');
            })
            .catch(function (err) {
                statusEl.textContent = 'Export failed: ' + err.message;
            })
            .finally(function () {
                exportBtn.disabled = false;
            });
    });
})();
