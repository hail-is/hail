#!/usr/bin/env python3
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DEPLOYMENT_FILES = [
    REPO_ROOT / 'gateway/deployment.yaml',
    REPO_ROOT / 'internal-gateway/deployment.yaml',
]
ENVOY_IMAGE_RE = re.compile(r'envoyproxy/envoy:v(?P<series>\d+\.\d+)\.(?P<patch>\d+)')


def current_envoy_version() -> tuple[str, str]:
    """Return (series, full_version) parsed from the first deployment file that contains an envoy image tag."""
    for f in DEPLOYMENT_FILES:
        if m := ENVOY_IMAGE_RE.search(f.read_text()):
            return m.group('series'), f"{m.group('series')}.{m.group('patch')}"
    raise SystemExit('ERROR: Could not find envoy version in deployment files')


def latest_patch_release(series: str) -> str:
    """Return the latest non-prerelease patch version within the given minor series from GitHub."""
    url = 'https://api.github.com/repos/envoyproxy/envoy/releases?per_page=50'
    req = urllib.request.Request(
        url,
        headers={'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'},
    )
    with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310

        releases = json.loads(r.read())

    series_re = re.compile(rf'^v{re.escape(series)}\.(\d+)$')
    patches = [
        (int(m.group(1)), release['tag_name'])
        for release in releases
        if (m := series_re.match(release['tag_name'])) and not release['prerelease']
    ]
    if not patches:
        raise SystemExit(f'ERROR: No releases found for series {series}')

    _, latest_tag = max(patches)
    return latest_tag.lstrip('v')


def update_files(old_version: str, new_version: str) -> None:
    """Replace the envoy image tag in all deployment files."""
    old_image = f'envoyproxy/envoy:v{old_version}'
    new_image = f'envoyproxy/envoy:v{new_version}'
    for f in DEPLOYMENT_FILES:
        updated = f.read_text().replace(old_image, new_image)
        f.write_text(updated)
        print(f'  Updated {f.relative_to(REPO_ROOT)}')


def main() -> None:
    series, current_version = current_envoy_version()
    print(f'Current version: v{current_version} (series {series})')
    print(f'Querying GitHub for latest v{series}.x release...')

    latest_version = latest_patch_release(series)

    if current_version == latest_version:
        print(f'Already up to date: v{current_version}')
        sys.exit(0)

    print(f'Updating v{current_version} -> v{latest_version}')
    update_files(current_version, latest_version)


if __name__ == '__main__':
    main()
