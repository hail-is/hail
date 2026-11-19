#!/usr/bin/env python3
"""Set, rotate or remove the secret CI uses to verify GitHub webhook deliveries.

Default (set / rotate):
  1. Check gh auth and the kubectl context
  2. Find CI's webhooks: repos from CI's watched branches, hooks whose URL is CI's /github_callback
  3. Generate a new secret and store it in the ci-github-webhook-secret k8s secret  (confirm before)
  4. Restart CI and wait for the rollout  (confirm before)
  5. Set the secret on every matched hook  (confirm before)
  6. Ping every hook and show the result

--unset (rollback to unverified deliveries):
  1-2 as above, then clear the secret on every hook, delete the k8s secret, restart CI, ping.

GitHub always follows CI: it gets a secret only after CI has it, and loses it before CI does.
The secret is never printed or written to disk. See dev-docs/services/ci/github-webhook-secret.md.
"""

import argparse
import base64
import itertools
import json
import secrets
import subprocess
import sys
import threading
import time

# GitHub webhooks only ever point at the default namespace's CI.
NAMESPACE = 'default'
K8S_SECRET = 'ci-github-webhook-secret'
K8S_SECRET_KEY = 'secret'


def run(args, input=None):  # pylint: disable=redefined-builtin
    result = subprocess.run(args, check=False, capture_output=True, text=True, input=input)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f'Command failed: {args}')
    return result.stdout


def spin(name, getter):
    """Run getter() while displaying a spinner, then print the result string it returns."""
    stop = threading.Event()
    value = None
    error: BaseException | None = None
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
        value = getter()
    except Exception as e:
        error = e
    finally:
        stop.set()
        t.join()

    if error is not None:
        print(f'\r{prefix}FAILED{"":<50}')
        raise error

    line = f'{prefix}{value}'
    print(f'\r{line:<60}')
    return value


def confirm(prompt):
    """Ask for y/N confirmation; exit on anything other than 'y'."""
    try:
        answer = input(f'{prompt} [y/N] ').strip().lower()
    except EOFError:
        answer = ''
    if answer != 'y':
        print('Aborted.')
        sys.exit(0)


def fail(message):
    print(f'\n{message}', file=sys.stderr)
    sys.exit(1)


def gh_api(path, method='GET', body=None):
    args = ['gh', 'api', '-X', method, '-H', 'Accept: application/vnd.github+json', path]
    if body is not None:
        args += ['--input', '-']
    out = run(args, input=json.dumps(body) if body is not None else None)
    return json.loads(out) if out.strip() else None


def kubectl(namespace, *args, input=None):  # pylint: disable=redefined-builtin
    return run(['kubectl', '--namespace', namespace, *args], input=input)


def read_k8s_secret(namespace, name, key):
    out = kubectl(namespace, 'get', 'secret', name, '-o', 'json')
    return base64.b64decode(json.loads(out)['data'][key]).decode()


def check_prereqs(namespace):
    print('Checking prerequisites...')

    def get_gh_user():
        result = subprocess.run(
            ['gh', 'auth', 'status', '--hostname', 'github.com'], check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or 'not authenticated')
        return run(['gh', 'api', 'user', '--jq', '.login']).strip()

    try:
        spin('  gh account      ', get_gh_user)
    except RuntimeError as e:
        fail(f'gh auth check failed: {e}\nRun: gh auth login')

    try:
        spin('  kubectl context ', lambda: run(['kubectl', 'config', 'current-context']).strip())
    except RuntimeError as e:
        fail(f'kubectl context check failed: {e}')
    print(f'  namespace       : {namespace}')
    print()


def discover(namespace):
    """Return (callback_url, [(repo, hook)]) for every hook pointing at this CI."""
    print('Discovering webhooks...')

    try:
        domain = spin('  domain          ', lambda: read_k8s_secret(namespace, 'global-config', 'domain'))
        watched = spin(
            '  watched branches', lambda: json.loads(read_k8s_secret(namespace, 'ci-config', 'watched_branches'))
        )
    except (RuntimeError, KeyError) as e:
        fail(f'Could not read CI config from namespace {namespace}: {e!r}')

    callback_url = f'https://ci.{domain}/github_callback'
    print(f'  callback URL    : {callback_url}')

    # watched_branches entries are [owner/repo:branch, deployable, mergeable]
    repos = sorted({branch.split(':', 1)[0] for branch, _, _ in watched})

    matched = []
    for repo in repos:
        try:
            hooks = spin(f'  hooks in {repo}', lambda repo=repo: gh_api(f'repos/{repo}/hooks?per_page=100'))
        except RuntimeError as e:
            fail(f'Could not list hooks for {repo} (needs repo admin): {e}')
        for hook in hooks:
            if hook['config'].get('url') == callback_url:
                matched.append((repo, hook))

    if not matched:
        fail(f'No webhooks point at {callback_url}. Nothing to do.')

    print()
    print('Matched webhooks:')
    for repo, hook in matched:
        has_secret = 'set' if hook['config'].get('secret') else 'none'
        print(f'  {repo}  hook {hook["id"]}  events: {",".join(hook["events"])}  secret: {has_secret}')
    print()
    return callback_url, matched


def restart_ci(namespace):
    def restart():
        kubectl(namespace, 'rollout', 'restart', 'deployment/ci')
        kubectl(namespace, 'rollout', 'status', 'deployment/ci', '--timeout=10m')
        return 'done'

    try:
        spin('  Restarting CI', restart)
    except RuntimeError as e:
        fail(f'CI restart failed: {e}\nCheck: kubectl -n {namespace} rollout status deployment/ci')


def set_hook_secret(matched, secret):
    for repo, hook in matched:
        path = f'repos/{repo}/hooks/{hook["id"]}/config'

        def patch(path=path):
            gh_api(path, method='PATCH', body={'secret': secret})
            # GitHub masks a set secret as '********' and omits it when there is none.
            now_set = bool(gh_api(path).get('secret'))
            if now_set != bool(secret):
                raise RuntimeError(f'hook secret is {"set" if now_set else "not set"} after the update')
            return 'done'

        try:
            spin(f'  {repo} hook {hook["id"]}', patch)
        except RuntimeError as e:
            fail(f'Updating {repo} hook {hook["id"]} failed: {e}\nRerun this script, or run it with --unset.')


def latest_ping(repo, hook):
    deliveries = gh_api(f'repos/{repo}/hooks/{hook["id"]}/deliveries?per_page=30')
    return next((d for d in deliveries if d['event'] == 'ping'), None)


def ping_hooks(matched, expected_status):
    all_ok = True
    sent = []
    for repo, hook in matched:
        try:
            # Remember the previous ping so an old one can't pass for this run's.
            previous = latest_ping(repo, hook)
            gh_api(f'repos/{repo}/hooks/{hook["id"]}/pings', method='POST')
        except RuntimeError as e:
            print(f'  {repo} hook {hook["id"]}: ping failed to send: {e}')
            all_ok = False
            continue
        sent.append((repo, hook, previous['id'] if previous else None))
    # Deliveries show up asynchronously; one short wait, no polling.
    time.sleep(5)

    for repo, hook, previous_id in sent:
        try:
            latest = latest_ping(repo, hook)
        except RuntimeError as e:
            print(f'  {repo} hook {hook["id"]}: could not list deliveries: {e}')
            all_ok = False
            continue
        if latest is None or latest['id'] == previous_id:
            print(f'  {repo} hook {hook["id"]}: no new ping delivery found yet')
            all_ok = False
            continue
        ok = latest['status_code'] == expected_status
        all_ok = all_ok and ok
        print(
            f'  {repo} hook {hook["id"]}: latest ping at {latest["delivered_at"]} -> '
            f'{latest["status_code"]} {latest["status"]}  {"OK" if ok else f"expected {expected_status}"}'
        )

    if not all_ok:
        print()
        print("Ping check didn't pass. Check CI's events page for github_webhook_rejected and the hook's")
        print('Recent Deliveries tab on GitHub. Missed deliveries are harmless: CI polls GitHub every 300s.')


def set_or_rotate(namespace):
    _, matched = discover(namespace)
    secret = secrets.token_hex(32)

    print(f'Step 1/4: Store a new secret in k8s secret {namespace}/{K8S_SECRET}')
    confirm('  Write k8s secret?')

    def apply_secret():
        manifest = kubectl(
            namespace,
            'create',
            'secret',
            'generic',
            K8S_SECRET,
            f'--from-file={K8S_SECRET_KEY}=/dev/stdin',
            '--dry-run=client',
            '-o',
            'json',
            input=secret,
        )
        try:
            kubectl(namespace, 'apply', '-f', '-', input=manifest)
        except RuntimeError as e:
            # kubectl apply errors can echo the patch body, which holds the secret.
            message = str(e)
            for s in (secret, base64.b64encode(secret.encode()).decode()):
                message = message.replace(s, '<redacted>')
            raise RuntimeError(message) from None
        return 'done'

    try:
        spin(f'  Applying {K8S_SECRET}', apply_secret)
    except RuntimeError as e:
        fail(f'Writing the k8s secret failed: {e}')
    print()

    print('Step 2/4: Restart CI so it picks up the secret')
    print('  From here until step 3 finishes, GitHub deliveries are rejected (CI polling covers the gap).')
    confirm('  Restart CI?')
    restart_ci(namespace)
    print()

    print(f'Step 3/4: Set the secret on {len(matched)} webhook(s)')
    confirm('  Update webhooks?')
    set_hook_secret(matched, secret)
    print()

    print('Step 4/4: Ping webhooks (expect 200)')
    ping_hooks(matched, expected_status=200)
    print()
    print('Done.')


def unset(namespace):
    _, matched = discover(namespace)

    print(f'Step 1/4: Clear the secret on {len(matched)} webhook(s)')
    print('  From here until step 3 finishes, GitHub deliveries are rejected (CI polling covers the gap).')
    confirm('  Clear webhook secrets?')
    set_hook_secret(matched, '')
    print()

    print(f'Step 2/4: Delete k8s secret {namespace}/{K8S_SECRET}')
    confirm('  Delete k8s secret?')

    def delete_secret():
        kubectl(namespace, 'delete', 'secret', K8S_SECRET, '--ignore-not-found')
        return 'done'

    try:
        spin(f'  Deleting {K8S_SECRET}', delete_secret)
    except RuntimeError as e:
        fail(f'Deleting the k8s secret failed: {e}')
    print()

    print('Step 3/4: Restart CI so it stops verifying')
    confirm('  Restart CI?')
    restart_ci(namespace)
    print()

    print('Step 4/4: Ping webhooks (expect 200, unsigned)')
    ping_hooks(matched, expected_status=200)
    print()
    print('Done. Webhook deliveries are no longer verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--unset', action='store_true', help='remove the secret: back to unverified deliveries')
    args = parser.parse_args()

    print(f'CI GitHub webhook secret: {"unset" if args.unset else "set / rotate"}')
    print()
    check_prereqs(NAMESPACE)
    if args.unset:
        unset(NAMESPACE)
    else:
        set_or_rotate(NAMESPACE)


if __name__ == '__main__':
    main()
