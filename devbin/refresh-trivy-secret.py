#!/usr/bin/env python3
"""Rotate the trivy-scanner GSA key and update the GOOGLE_GAR_CREDENTIALS GitHub secret.

Full procedure:
  1. Verify gcloud and gh authentication
  2. List existing user-managed keys for the trivy-scanner service account
  3. Create a new GSA key  (confirm before)
  4. Update the GOOGLE_GAR_CREDENTIALS secret in the hail-is/hail GitHub repo  (confirm before)
  5. Delete the old key(s)  (confirm before)
  6. Clean up the temporary key file

See dev-docs/refreshing-trivy-scanner-github-secret.md for context.
"""
import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

PROJECT = 'hail-vdc'
SA_NAME = 'trivy-scanner'
SA_EMAIL = f'{SA_NAME}@{PROJECT}.iam.gserviceaccount.com'
GITHUB_REPO = 'hail-is/hail'
GITHUB_SECRET = 'GOOGLE_GAR_CREDENTIALS'


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f'Command failed: {args}')
    return result.stdout


def spin(name, getter):
    """Run getter() while displaying a spinner, then print the result string it returns."""
    stop = threading.Event()
    result = [None]
    error = [None]
    prefix = f'{name}: '

    def _spin():
        for frame in itertools.cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'):
            if stop.is_set():
                break
            print(f'\r{prefix}{frame}', end='', flush=True)
            time.sleep(0.1)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        result[0] = getter()
    except Exception as e:
        error[0] = e
    finally:
        stop.set()
        t.join()

    if error[0] is not None:
        print(f'\r{prefix}FAILED{"":<50}')
        raise error[0]

    line = f'{prefix}{result[0]}'
    print(f'\r{line:<60}')
    return result[0]


def confirm(prompt):
    """Ask for y/N confirmation; exit on anything other than 'y'."""
    try:
        answer = input(f'{prompt} [y/N] ').strip().lower()
    except EOFError:
        answer = ''
    if answer != 'y':
        print('Aborted.')
        sys.exit(0)


def check_prereqs():
    """Verify gcloud and gh are authenticated; print who we're acting as."""
    print('Checking prerequisites...')

    def get_gcloud_account():
        out = run(['gcloud', 'auth', 'list', '--filter', 'status=ACTIVE', '--format', 'value(account)'])
        account = out.strip()
        if not account:
            raise RuntimeError('No active gcloud account found')
        return account

    try:
        spin('  gcloud account  ', get_gcloud_account)
    except RuntimeError as e:
        print(f'\ngcloud auth check failed: {e}', file=sys.stderr)
        print('Run: gcloud auth login', file=sys.stderr)
        sys.exit(1)

    def get_gh_user():
        result = subprocess.run(['gh', 'auth', 'status', '--hostname', 'github.com'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or 'not authenticated')
        user_result = subprocess.run(['gh', 'api', 'user', '--jq', '.login'], capture_output=True, text=True)
        return user_result.stdout.strip() if user_result.returncode == 0 else '(authenticated)'

    try:
        spin('  gh account      ', get_gh_user)
    except RuntimeError as e:
        print(f'\ngh auth check failed: {e}', file=sys.stderr)
        print('Run: gh auth login', file=sys.stderr)
        sys.exit(1)

    print()


def list_keys():
    out = run([
        'gcloud', 'iam', 'service-accounts', 'keys', 'list',
        '--iam-account', SA_EMAIL,
        '--project', PROJECT,
        '--managed-by', 'user',
        '--format', 'json',
    ])
    return json.loads(out)


def main():
    print('Trivy scanner GSA key rotation')
    print(f'  Service account : {SA_EMAIL}')
    print(f'  GitHub secret   : {GITHUB_REPO} / {GITHUB_SECRET}')
    print()

    check_prereqs()

    existing_keys = []

    def fetch_keys():
        existing_keys.extend(list_keys())
        n = len(existing_keys)
        return f'{n} key{"s" if n != 1 else ""} found'

    try:
        spin('  Fetching existing keys', fetch_keys)
    except RuntimeError as e:
        print(f'\nError fetching keys: {e}', file=sys.stderr)
        sys.exit(1)

    if not existing_keys:
        print('  Warning: no existing user-managed keys found — the account may already be clean.')
    else:
        for k in existing_keys:
            key_id = k['name'].split('/')[-1]
            print(f'    {key_id}  created: {k["validAfterTime"]}')

    print()

    key_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            key_file = f.name

        # --- Step 1: create new key ---
        print(f'Step 1/3: Create a new GSA key for {SA_EMAIL}')
        confirm('  Create new key?')

        def create_key():
            run([
                'gcloud', 'iam', 'service-accounts', 'keys', 'create',
                key_file,
                '--iam-account', SA_EMAIL,
                '--project', PROJECT,
            ])
            return 'done'

        try:
            spin('  Creating new GSA key', create_key)
        except RuntimeError as e:
            print(f'\nError creating GSA key: {e}', file=sys.stderr)
            sys.exit(1)

        with open(key_file) as f:
            key_json = f.read()
        new_key_id = json.loads(key_json).get('private_key_id', '(unknown)')
        print(f'  New key ID: {new_key_id}')
        print()

        # --- Step 2: update GitHub secret ---
        print(f'Step 2/3: Update {GITHUB_SECRET} in {GITHUB_REPO}')
        confirm('  Update GitHub secret?')

        def update_github_secret():
            run(['gh', 'secret', 'set', GITHUB_SECRET, '--repo', GITHUB_REPO, '--body', key_json])
            return 'done'

        try:
            spin(f'  Updating {GITHUB_SECRET}', update_github_secret)
        except RuntimeError as e:
            print(f'\nError updating GitHub secret: {e}', file=sys.stderr)
            print(
                'The new GSA key was created but the GitHub secret was NOT updated.\n'
                'Manual follow-up required — see dev-docs/refreshing-trivy-scanner-github-secret.md',
                file=sys.stderr,
            )
            sys.exit(1)

        print()

        # --- Step 3: delete old keys ---
        if existing_keys:
            print(f'Step 3/3: Delete {len(existing_keys)} old key(s)')
            for k in existing_keys:
                key_id = k['name'].split('/')[-1]
                print(f'  {key_id}  created: {k["validAfterTime"]}')
            confirm('  Delete these keys?')

            def delete_old_keys():
                for k in existing_keys:
                    key_id = k['name'].split('/')[-1]
                    run([
                        'gcloud', 'iam', 'service-accounts', 'keys', 'delete', key_id,
                        '--iam-account', SA_EMAIL,
                        '--project', PROJECT,
                        '--quiet',
                    ])
                return f'{len(existing_keys)} deleted'

            try:
                spin('  Deleting old key(s)', delete_old_keys)
            except RuntimeError as e:
                print(f'\nError deleting old key(s): {e}', file=sys.stderr)
                print('The GitHub secret was updated successfully, but old key(s) may still exist.', file=sys.stderr)
                print('Clean them up manually in the GCP console.', file=sys.stderr)
                sys.exit(1)
        else:
            print('Step 3/3: No old keys to delete.')

        print()

    finally:
        if key_file and os.path.exists(key_file):
            os.unlink(key_file)

    print('Done. Verify by running the Trivy Security Scan workflow:')
    print('  gh workflow run trivy-scan.yml --repo hail-is/hail --ref main')


if __name__ == '__main__':
    main()
