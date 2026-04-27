(function () {
    var exportBtn = document.getElementById('billing-export-btn');
    var copyBtn = document.getElementById('billing-copy-btn');
    if (!exportBtn && !copyBtn) return;

    var statusEl = document.getElementById('billing-export-status');
    var basePath = (document.getElementById('billing-js') || {}).dataset?.basePath ?? '';

    // Converts a mm/dd/yyyy query-param date string to yyyy-mm-dd for use in filenames.
    // String manipulation is intentional — Date.toISOString() converts to UTC and can
    // shift the date backwards in timezones behind UTC.
    function queryDateToIso8601(mmddyyyy) {
        try {
            if (!mmddyyyy) {
                const now = new Date();
                yyyy = now.getFullYear();
                mm = String(now.getMonth() + 1).padStart(2, '0');
                dd = String(now.getDate()).padStart(2, '0');
                return `${yyyy}-${mm}-${dd}`;
            } else {
                const [mm, dd, yyyy] = mmddyyyy.split('/');
                return `${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`;
            }
        } catch (e) {
            return 'invalid-date-format';
        }
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
        rows.forEach(function (row) {
            lines.push(columns.map(function (col) { return csvEscape(row.get(col)); }).join(','));
        });
        return lines.join('\n');
    }

    function groupByProject(records) {
        var acc = new Map();
        records.forEach(function (r) {
            acc.set(r.billing_project, (acc.get(r.billing_project) || 0) + r.total_spent);
        });
        return Array.from(acc.keys()).sort().map(function (bp) {
            return new Map([['billing_project', bp], ['total_spent', acc.get(bp)]]);
        });
    }

    function groupByUser(records) {
        var acc = new Map();
        records.forEach(function (r) {
            acc.set(r.user, (acc.get(r.user) || 0) + r.total_spent);
        });
        return Array.from(acc.keys()).sort().map(function (u) {
            return new Map([['user', u], ['total_spent', acc.get(u)]]);
        });
    }

    function groupByProjectUser(records) {
        return records.slice().sort(function (a, b) {
            if (a.billing_project !== b.billing_project) {
                return a.billing_project < b.billing_project ? -1 : 1;
            }
            return a.user < b.user ? -1 : 1;
        }).map(function (r) {
            return new Map([['billing_project', r.billing_project], ['user', r.user], ['total_spent', r.total_spent]]);
        });
    }

    function getGrouping() {
        var radios = document.querySelectorAll('input[name="export-grouping"]');
        var checked = Array.from(radios).find(function (r) { return r.checked; });
        return checked ? checked.value : 'by-project-user';
    }

    function prepareCsvExport(records) {
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
        var startStr = queryDateToIso8601(params.get('start') || '');
        var endStr = queryDateToIso8601(params.get('end') || '');
        return {
            csvText: toCsv(rows, columns),
            filename: `Hail billing export ${startStr} to ${endStr} ${label}.csv`,
        };
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

    function copyToClipboard(csvText) {
        return navigator.clipboard.writeText(csvText);
    }

    function fetchBillingRecords() {
        return fetch(basePath + '/api/v1alpha/billing' + window.location.search)
            .then(function (resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json();
            });
    }

    function withBillingData(btn, action) {
        btn.disabled = true;
        statusEl.textContent = 'Fetching\u2026';
        statusEl.classList.remove('hidden');

        fetchBillingRecords()
            .then(function (records) { return action(prepareCsvExport(records)); })
            .catch(function (err) {
                statusEl.textContent = 'Failed: ' + err.message;
            })
            .finally(function () {
                btn.disabled = false;
            });
    }

    if (exportBtn) {
        exportBtn.addEventListener('click', function () {
            withBillingData(exportBtn, function ({ csvText, filename }) {
                triggerDownload(csvText, filename);
                statusEl.classList.add('hidden');
            });
        });
    }

    if (copyBtn) {
        copyBtn.addEventListener('click', function () {
            withBillingData(copyBtn, function ({ csvText }) {
                return copyToClipboard(csvText).then(function () {
                    statusEl.classList.add('hidden');
                    var original = copyBtn.textContent;
                    copyBtn.textContent = '\u2713 Copied!';
                    setTimeout(function () { copyBtn.textContent = original; }, 2000);
                });
            });
        });
    }
})();
