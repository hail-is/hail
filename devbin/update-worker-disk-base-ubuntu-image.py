#!/usr/bin/env python3
"""Prepare files for a batch worker VM image update and optionally kick off a test build.

Sets up for a test build in your personal namespace. After the test succeeds,
run the create script again with NAMESPACE=default and update create_instance.py
to the production image name before merging.

Updates:
  - UBUNTU_IMAGE in batch/gcp-create-worker-image.sh (latest Noble minimal from GCP)
  - WORKER_IMAGE_VERSION in batch/gcp-create-worker-image.sh (incremented by 1)
  - batch-worker-N image reference in batch/batch/cloud/gcp/driver/create_instance.py
    (set to the personal-namespace test image name)
  - INSTANCE_VERSION in batch/batch/globals.py (incremented by 1)
"""

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CREATE_SCRIPT = REPO_ROOT / 'batch/gcp-create-worker-image.sh'
STARTUP_SCRIPT = REPO_ROOT / 'batch/build-batch-worker-image-startup-gcp.sh'
CREATE_INSTANCE_PY = REPO_ROOT / 'batch/batch/cloud/gcp/driver/create_instance.py'
GLOBALS_PY = REPO_ROOT / 'batch/batch/globals.py'

NVIDIA_DOWNLOAD_INDEX = 'https://download.nvidia.com/XFree86/Linux-x86_64/'


def ask(prompt: str) -> bool:
    response = input(f'{prompt} [y/N] ').strip().lower()
    return response in ('y', 'yes')


def latest_ubuntu_noble_image() -> str:
    print('Querying GCP for latest ubuntu-minimal noble amd64 image...')
    result = subprocess.run(
        [
            'gcloud', 'compute', 'images', 'list',
            '--project', 'ubuntu-os-cloud',
            '--filter', 'name~^ubuntu-minimal-2404-noble-amd64-v',
            '--sort-by', '~creationTimestamp',
            '--limit', '1',
            '--format', 'value(name)',
            '--no-standard-images',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    image = result.stdout.strip()
    if not image:
        print('ERROR: No ubuntu-minimal-2404-noble-amd64 image found', file=sys.stderr)
        sys.exit(1)
    return image


def update_create_script(new_ubuntu_image: str) -> tuple[int, int]:
    """Update UBUNTU_IMAGE and WORKER_IMAGE_VERSION. Returns (old_version, new_version)."""
    text = CREATE_SCRIPT.read_text()

    ubuntu_m = re.search(r'^(UBUNTU_IMAGE=)(ubuntu-minimal-2404-noble-amd64-v\S+)$', text, re.MULTILINE)
    if not ubuntu_m:
        print('ERROR: Could not find UBUNTU_IMAGE= line in create script', file=sys.stderr)
        sys.exit(1)
    current_ubuntu = ubuntu_m.group(2)

    version_m = re.search(r'^(WORKER_IMAGE_VERSION=)(\d+)$', text, re.MULTILINE)
    if not version_m:
        print('ERROR: Could not find WORKER_IMAGE_VERSION= line in create script', file=sys.stderr)
        sys.exit(1)
    old_version = int(version_m.group(2))
    new_version = old_version + 1

    text = text.replace(f'UBUNTU_IMAGE={current_ubuntu}', f'UBUNTU_IMAGE={new_ubuntu_image}')
    text = text.replace(f'WORKER_IMAGE_VERSION={old_version}', f'WORKER_IMAGE_VERSION={new_version}')
    CREATE_SCRIPT.write_text(text)

    print(f'  {CREATE_SCRIPT.relative_to(REPO_ROOT)}: UBUNTU_IMAGE {current_ubuntu} -> {new_ubuntu_image}')
    print(f'  {CREATE_SCRIPT.relative_to(REPO_ROOT)}: WORKER_IMAGE_VERSION {old_version} -> {new_version}')
    return old_version, new_version


def update_create_instance(old_version: int, new_ref: str) -> None:
    text = CREATE_INSTANCE_PY.read_text()
    old_ref = f'batch-worker-{old_version}'
    if old_ref not in text:
        print(f'ERROR: Could not find {old_ref!r} in {CREATE_INSTANCE_PY}', file=sys.stderr)
        sys.exit(1)
    CREATE_INSTANCE_PY.write_text(text.replace(old_ref, new_ref))
    print(f'  {CREATE_INSTANCE_PY.relative_to(REPO_ROOT)}: {old_ref} -> {new_ref}')


def update_globals() -> None:
    text = GLOBALS_PY.read_text()
    m = re.search(r'^(INSTANCE_VERSION\s*=\s*)(\d+)$', text, re.MULTILINE)
    if not m:
        print('ERROR: Could not find INSTANCE_VERSION in globals.py', file=sys.stderr)
        sys.exit(1)
    old = int(m.group(2))
    new = old + 1
    GLOBALS_PY.write_text(text.replace(f'{m.group(1)}{old}', f'{m.group(1)}{new}'))
    print(f'  {GLOBALS_PY.relative_to(REPO_ROOT)}: INSTANCE_VERSION {old} -> {new}')


def current_nvidia_version() -> str:
    text = STARTUP_SCRIPT.read_text()
    m = re.search(r'NVIDIA-Linux-x86_64-(\d+\.\d+\.\d+)\.run', text)
    if not m:
        return 'unknown'
    return m.group(1)


def nvidia_sha256(version: str) -> str:
    """Fetch the published sha256 for the given NVIDIA driver .run file."""
    url = f'{NVIDIA_DOWNLOAD_INDEX}{version}/NVIDIA-Linux-x86_64-{version}.run.sha256sum'
    with urllib.request.urlopen(url, timeout=15) as r:  # nosec B310
        return r.read().decode().split()[0]


def update_nvidia_driver(new_version: str) -> None:
    """Replace the NVIDIA driver version and sha256 in the startup script."""
    old_version = current_nvidia_version()
    text = STARTUP_SCRIPT.read_text()

    old_sha_m = re.search(r'([0-9a-f]{64})  NVIDIA-Linux-x86_64-' + re.escape(old_version) + r'\.run', text)
    if not old_sha_m:
        print(f'  ERROR: Could not find sha256 for {old_version} in startup script', file=sys.stderr)
        sys.exit(1)
    old_sha256 = old_sha_m.group(1)

    print(f'  Fetching sha256 for NVIDIA {new_version}...')
    new_sha256 = nvidia_sha256(new_version)

    text = text.replace(old_version, new_version)
    text = text.replace(old_sha256, new_sha256)

    STARTUP_SCRIPT.write_text(text)
    print(f'  {STARTUP_SCRIPT.relative_to(REPO_ROOT)}: NVIDIA driver {old_version} -> {new_version}')
    print(f'  {STARTUP_SCRIPT.relative_to(REPO_ROOT)}: sha256 updated')


def nvidia_min_kernel(version: str) -> str:
    """Return the minimum kernel requirement for the given NVIDIA driver version."""
    url = f'{NVIDIA_DOWNLOAD_INDEX}{version}/README/minimumrequirements.html'
    try:
        with urllib.request.urlopen(url, timeout=10) as r:  # nosec B310
            html = r.read().decode()
        m = re.search(r'Linux kernel</td>\s*<td>([\d.]+) and newer', html)
        return m.group(1) if m else '?'
    except Exception:
        return '?'


def available_nvidia_versions() -> list[str]:
    """Return all driver versions listed in the NVIDIA download index."""
    with urllib.request.urlopen(NVIDIA_DOWNLOAD_INDEX, timeout=15) as r:  # nosec B310
        html = r.read().decode()
    versions = re.findall(r'href=["\'](\d{3,}\.\d+\.\d+)/["\']', html)
    return versions


def nvidia_release_dates(versions: list[str]) -> dict[str, str]:
    """Return {version: YYYY-MM-DD} for the given versions, looked up from GitHub releases."""
    import json

    req = urllib.request.Request(
        'https://api.github.com/repos/NVIDIA/open-gpu-kernel-modules/releases?per_page=100',
        headers={'Accept': 'application/vnd.github+json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:  # nosec B310
            releases = json.loads(r.read())
    except Exception:
        return {}

    version_set = set(versions)
    return {
        rel['tag_name']: rel['published_at'][:10]
        for rel in releases
        if rel['tag_name'] in version_set
    }


def ver_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split('.'))


def show_nvidia_info() -> str | None:
    """Print NVIDIA driver info. Returns the suggested new version if an update is available."""
    print()
    print('--- NVIDIA Driver Check ---')
    current = current_nvidia_version()
    series = current.split('.')[0]
    print(f'  Current driver in startup script: {current}')

    print('  Fetching available NVIDIA driver versions...')
    try:
        all_versions = available_nvidia_versions()
    except Exception as e:
        print(f'  WARNING: Could not fetch NVIDIA version list: {e}')
        print(f'  Check manually: {NVIDIA_DOWNLOAD_INDEX}')
        return None

    same_series = [v for v in all_versions if v.startswith(f'{series}.')]
    latest_in_series = max(same_series, key=ver_tuple) if same_series else None
    latest_overall = max(all_versions, key=ver_tuple) if all_versions else None

    versions_to_check = {current}
    if latest_in_series:
        versions_to_check.add(latest_in_series)
    if latest_overall:
        versions_to_check.add(latest_overall)

    min_kernels: dict[str, str] = {}
    for v in versions_to_check:
        min_kernels[v] = nvidia_min_kernel(v)

    release_dates = nvidia_release_dates(list(versions_to_check))

    def fmt(v: str, label: str) -> str:
        date = release_dates.get(v, '?')
        update = '  <-- update available' if v != current else ''
        # NVIDIA publishes no formal max kernel; release date is the best proxy
        return f'  {label}: {v}  (released: {date}, min kernel: {min_kernels.get(v, "?")}){update}'

    print(fmt(current, f'Current ({series}.x)'))
    if latest_in_series and latest_in_series != current:
        print(fmt(latest_in_series, f'Latest {series}.x '))
    else:
        print(f'  Latest {series}.x:  {latest_in_series}  (up to date)')
    if latest_overall and latest_overall != latest_in_series:
        print(fmt(latest_overall, 'Latest overall '))

    print()
    print('  Note: NVIDIA does not publish a maximum supported kernel. The release date is')
    print('  the best proxy — a driver released before a kernel came out may not compile against it.')
    print('  Ubuntu 24.04 Noble ships with kernel 6.8.x (HWE variants may be newer).')
    print('  A driver too old for the running kernel causes the image build to hang silently.')

    # Prefer latest in series (safer); fall back to latest overall if no series update
    suggested = latest_in_series if (latest_in_series and latest_in_series != current) else None
    if suggested is None and latest_overall and latest_overall != current:
        suggested = latest_overall
    return suggested



def print_next_steps(namespace: str, new_worker_version: int) -> None:
    test_image = f'batch-worker-{namespace}-{new_worker_version}'
    prod_image = f'batch-worker-{new_worker_version}'

    print()
    print('=' * 60)
    print('Remaining steps:')
    print('=' * 60)
    print()
    print('1. If you did not start the test build above, run it now:')
    print(f'     NAMESPACE={namespace} batch/gcp-create-worker-image.sh')
    print()
    print('2. Monitor the build:')
    print(f'   - Open the GCP Console VM list and find the build-batch-worker-{namespace}-image VM.')
    print('   - Under the three-dot menu, open "Monitoring" to watch serial port logs.')
    print('   - If the build stalls for more than ~15 minutes, check the logs for')
    print('     NVIDIA driver errors (a common cause of silent hangs).')
    print()
    print('3. Once the test build succeeds, build the production image:')
    print('     NAMESPACE=default batch/gcp-create-worker-image.sh')
    print(f'   Then update create_instance.py: {test_image} -> {prod_image}')
    print()
    print('4. Create a PR with these file changes, run CI, and deploy.')
    print('   Note: the rollout only affects newly created workers.')
    print('   Existing workers stay on the old image until deleted.')


def main() -> None:
    namespace = os.environ.get('NAMESPACE', '').strip()
    if not namespace:
        namespace = input('Personal namespace for test build (e.g. your name): ').strip()
    if not namespace:
        print('ERROR: A personal namespace is required.', file=sys.stderr)
        sys.exit(1)
    if namespace == 'default':
        print("ERROR: NAMESPACE must be a personal namespace, not 'default'.", file=sys.stderr)
        sys.exit(1)

    if not shutil.which('gcloud'):
        print('ERROR: gcloud is required but not found in PATH.', file=sys.stderr)
        sys.exit(1)

    new_ubuntu = latest_ubuntu_noble_image()
    print('Updating files...')
    old_version, new_version = update_create_script(new_ubuntu)
    test_image = f'batch-worker-{namespace}-{new_version}'
    update_create_instance(old_version, test_image)
    update_globals()

    suggested_nvidia = show_nvidia_info()

    if suggested_nvidia:
        print()
        if ask(f'Update NVIDIA driver to {suggested_nvidia}?'):
            update_nvidia_driver(suggested_nvidia)

    print()
    cmd = f'NAMESPACE={namespace} {CREATE_SCRIPT.relative_to(REPO_ROOT)}'
    if ask(f'About to run: {cmd}\nContinue?'):
        subprocess.run([str(CREATE_SCRIPT)], env={**os.environ, 'NAMESPACE': namespace}, check=True)

    print_next_steps(namespace, new_version)


if __name__ == '__main__':
    main()
