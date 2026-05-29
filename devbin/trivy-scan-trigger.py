#!/usr/bin/env python3
"""Find the most recent completed ci/deploy/prod batch and extract the batch image name."""
import itertools
import json
import re
import subprocess
import sys
import threading
import time


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def print_placeholder_during_lookup(name, getter):
    stop = threading.Event()
    result = [None]
    prefix = f'{name} : '

    def spin():
        for frame in itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'):
            if stop.is_set():
                break
            print(f'\r{prefix}{frame}', end='', flush=True)
            time.sleep(0.1)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        result[0] = getter()
    finally:
        stop.set()
        t.join()

    line = f'{prefix}{result[0]}'
    print(f'\r{line:<60}')
    return result[0]


def find_batch_id():
    out = run(['hailctl', 'batch', 'list', '--query', 'batch_type=ci/deploy/prod complete', '--limit', '1', '-o', 'json'])
    if not out.strip() or not out.strip().startswith('['):
        print('No completed ci/deploy/prod batches found', file=sys.stderr)
        sys.exit(1)
    batches = json.loads(out)
    if not batches:
        print('No completed ci/deploy/prod batches found', file=sys.stderr)
        sys.exit(1)
    return batches[0]['id']


def find_sha(batch_id):
    batch = json.loads(run(['hailctl', 'batch', 'get', str(batch_id), '-o', 'json']))[0]
    return batch['attributes']['sha']


def find_image(batch_id):
    out = run(['hailctl', 'batch', 'jobs', str(batch_id), '--name', 'batch_image', '-o', 'json'])
    if not out.strip() or not out.strip().startswith('['):
        print(f'No batch_image job found in batch {batch_id}', file=sys.stderr)
        sys.exit(1)
    jobs = json.loads(out)
    if not jobs:
        print(f'No batch_image job found in batch {batch_id}', file=sys.stderr)
        sys.exit(1)
    job_id = jobs[0]['job_id']
    job = json.loads(run(['hailctl', 'batch', 'job', str(batch_id), str(job_id), '-o', 'json']))[0]
    command_parts = job['spec']['process']['command']
    command = next((p for p in command_parts if '--output' in p), None)
    if command is None:
        print('No --output flag found in batch_image job command', file=sys.stderr)
        sys.exit(1)
    match = re.search(r"--output 'type=image,\"name=([^,\"]+)", command)
    if not match:
        print('Could not find image name in batch_image job command', file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def main():
    print('Searching latest completed deploy batch for commit sha and image to scan...')
    print()
    batch_id = print_placeholder_during_lookup('  Batch', find_batch_id)
    sha      = print_placeholder_during_lookup('  SHA  ', lambda: find_sha(batch_id))
    image    = print_placeholder_during_lookup('  Image', lambda: find_image(batch_id))

    BRANCH = 'main'
    gh_cmd = (
        f'gh workflow run trivy-scan.yml --repo hail-is/hail --ref {BRANCH}'
        f' -f branch={BRANCH} -f commit_hash={sha} -f images={image}'
    )
    gh_cmd_display = (
        f'gh workflow run trivy-scan.yml \\\n'
        f'    --repo hail-is/hail --ref {BRANCH} \\\n'
        f'    -f branch={BRANCH} \\\n'
        f'    -f commit_hash={sha} \\\n'
        f'    -f images={image}'
    )

    print()
    print(f'Proposed command:\n{gh_cmd_display}')
    print()
    try:
        answer = input('Run it now? [y/N] ').strip().lower()
    except EOFError:
        answer = ''
    if answer != 'y':
        print('Aborted.')
        sys.exit(0)

    subprocess.run(
        [
            'gh', 'workflow', 'run', 'trivy-scan.yml',
            '--repo', 'hail-is/hail',
            '--ref', BRANCH,
            '-f', f'branch={BRANCH}',
            '-f', f'commit_hash={sha}',
            '-f', f'images={image}',
        ],
        check=True,
    )
    print('Workflow triggered.')


if __name__ == '__main__':
    main()
