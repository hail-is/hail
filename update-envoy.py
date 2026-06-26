#!/usr/bin/env python3
import re
import sys
import urllib.request
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent
FILES = [
    REPO_ROOT / 'gateway/deployment.yaml',
    REPO_ROOT / 'internal-gateway/deployment.yaml',
]

pattern = re.compile(r'envoyproxy/envoy:v(\d+\.\d+)\.\d+')

current_version = None
for f in FILES:
    m = pattern.search(f.read_text())
    if m:
        current_version = m.group(0).split(':v')[1]
        current_series = m.group(1)
        break

if not current_version:
    print('ERROR: Could not find envoy version in deployment files', file=sys.stderr)
    sys.exit(1)

print(f'Current version: v{current_version} (series {current_series})')
print(f'Querying GitHub for latest v{current_series}.x release...')

url = f'https://api.github.com/repos/envoyproxy/envoy/releases?per_page=50'
req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
with urllib.request.urlopen(req) as r:  # nosec B310
    releases = json.loads(r.read())

series_pattern = re.compile(rf'^v{re.escape(current_series)}\.(\d+)$')
matching = [(int(m.group(1)), r['tag_name']) for r in releases if (m := series_pattern.match(r['tag_name'])) and not r['prerelease']]

if not matching:
    print(f'ERROR: No releases found for series {current_series}', file=sys.stderr)
    sys.exit(1)

latest_patch, latest_tag = max(matching)
latest_version = latest_tag.lstrip('v')

if current_version == latest_version:
    print(f'Already up to date: v{current_version}')
    sys.exit(0)

print(f'Updating v{current_version} -> v{latest_version}')
for f in FILES:
    text = f.read_text()
    updated = text.replace(f'envoyproxy/envoy:v{current_version}', f'envoyproxy/envoy:v{latest_version}')
    f.write_text(updated)
    print(f'  Updated {f.relative_to(REPO_ROOT)}')
