#!/usr/bin/env python3
"""Cloud NAT cost report: month-by-month, region-by-region inbound/outbound bytes and cost.

Usage:
    python nat_cost_report.py [project] > report.csv
    python nat_cost_report.py hail-vdc > nat_costs.csv
"""

import calendar
import csv
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import date


PROJECT = sys.argv[1] if len(sys.argv) > 1 else 'hail-vdc'

# GCP Cloud NAT data processing tiers (applied to total bytes per month, all regions combined)
TIERS = [
    (1_000, 0.045),       # first 1 TB
    (10_000, 0.020),      # 1–10 TB
    (float('inf'), 0.010),  # above 10 TB
]


def get_access_token():
    r = subprocess.run(['gcloud', 'auth', 'print-access-token'], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def query_nat_bytes(token, project, metric_suffix, start_iso, end_iso, alignment_secs):
    """Returns {region: bytes} for the given metric and time window."""
    params = {
        'filter': f'metric.type="router.googleapis.com/nat/{metric_suffix}"',
        'interval.startTime': start_iso,
        'interval.endTime': end_iso,
        'aggregation.alignmentPeriod': f'{alignment_secs}s',
        'aggregation.perSeriesAligner': 'ALIGN_DELTA',
        'aggregation.crossSeriesReducer': 'REDUCE_SUM',
        'aggregation.groupByFields': 'resource.region',
    }
    url = (
        f'https://monitoring.googleapis.com/v3/projects/{project}/timeSeries'
        f'?{urllib.parse.urlencode(params)}'
    )
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)

    result = {}
    for ts in data.get('timeSeries', []):
        region = ts['resource']['labels']['region']
        total = sum(int(p['value'].get('int64Value', 0)) for p in ts.get('points', []))
        result[region] = result.get(region, 0) + total
    return result


def tiered_cost(total_gb):
    cost = 0.0
    prev = 0.0
    for cap_gb, rate in TIERS:
        if total_gb <= prev:
            break
        chargeable = min(total_gb, cap_gb) - prev
        cost += chargeable * rate
        prev = cap_gb
    return cost


def months_to_query(n):
    """Yield (year, month, start, end) for the last n complete months plus the current partial month, oldest first."""
    today = date.today()
    months = [(today.year, today.month)]  # current partial month
    year, month = today.year, today.month
    for _ in range(n):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        months.append((year, month))
    for year, month in reversed(months):
        start = date(year, month, 1)
        if (year, month) == (today.year, today.month):
            end = today
        else:
            end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        yield year, month, start, end


def main():
    token = get_access_token()

    # Collect raw byte counts: {month: {region: {in: bytes, out: bytes}}}
    # Pre-initialize all months so they appear in output even if no data
    data_by_month = {}
    alignment_secs = 31 * 24 * 3600  # long enough to cover any month in one bucket
    all_regions = set()

    for year, month, start, end in months_to_query(12):
        label = start.strftime('%Y-%m')
        data_by_month[label] = {}  # ensure month appears even with no data
        start_iso = f'{start.isoformat()}T00:00:00Z'
        end_iso = f'{end.isoformat()}T00:00:00Z'

        print(f'Querying {label}...', file=sys.stderr)
        inbound = query_nat_bytes(token, PROJECT, 'received_bytes_count', start_iso, end_iso, alignment_secs)
        outbound = query_nat_bytes(token, PROJECT, 'sent_bytes_count', start_iso, end_iso, alignment_secs)

        regions = set(inbound) | set(outbound)
        all_regions.update(regions)
        data_by_month[label] = {r: {'in': inbound.get(r, 0), 'out': outbound.get(r, 0)} for r in regions}

    # Build one row per month; columns: month, <region>:inbound_gb, <region>:outbound_gb, ..., total_cost_usd
    regions = sorted(all_regions)
    fieldnames = ['month']
    for r in regions:
        fieldnames += [f'{r}:inbound_gb', f'{r}:outbound_gb']
    fieldnames.append('total_cost_usd')

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for label, region_data in data_by_month.items():
        total_bytes = sum(v['in'] + v['out'] for v in region_data.values())
        total_cost = tiered_cost(total_bytes / 1e9)

        row = {'month': label, 'total_cost_usd': round(total_cost, 2)}
        for r in regions:
            d = region_data.get(r, {'in': 0, 'out': 0})
            row[f'{r}:inbound_gb'] = round(d['in'] / 1e9, 3)
            row[f'{r}:outbound_gb'] = round(d['out'] / 1e9, 3)
        writer.writerow(row)


if __name__ == '__main__':
    main()
