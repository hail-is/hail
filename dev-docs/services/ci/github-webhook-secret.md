# CI GitHub webhook secret

GitHub sends CI events (pushes, PRs, reviews) to `POST /github_callback`. With a webhook secret, GitHub signs each
delivery (`X-Hub-Signature-256`, an HMAC-SHA256 of the body) and CI rejects any delivery whose signature doesn't
match, with a 401.

- CI reads the secret from the k8s secret `ci-github-webhook-secret` (key `secret`) at startup. It's optional.
- **Secret set:** every delivery must be signed with it.
- **No secret:** nothing is verified, and signed deliveries are rejected. CI logs
  `github webhook signature verification: DISABLED` at startup and, in `default`, records a
  `github_webhook_verification_disabled` event.
- Rejected deliveries are always logged. A bad signature also records a `github_webhook_rejected` event, at most one
  every 10 minutes.

Webhook problems are never urgent. CI polls GitHub every 300 seconds, so missed or rejected deliveries only make CI
slower to react.

> [!WARNING]
> GitHub must never have a secret that CI doesn't. Never set the secret on the webhook by hand: use the script, which
> updates CI first and GitHub second (and the reverse with `--unset`).

## When to rotate

Only when needed: if the secret may have leaked, or someone who had access to it (k8s secrets in `default`, or repo
admin on GitHub) leaves. There's no scheduled rotation.

## Setting or rotating the secret

You need `gh` logged in as a repo admin (managing webhooks needs Admin) and `kubectl` pointed at CI's cluster.

```bash
python3 devbin/refresh-ci-webhook-secret.py
```

The script:

1. Shows your `gh` user and kubectl context.
2. Finds CI's webhooks: the repos in CI's watched branches (`ci-config` secret), and in each repo the hooks whose URL
   is exactly `https://ci.<domain>/github_callback`. It lists them and stops if there are none.
3. Generates a new secret and writes it to `ci-github-webhook-secret`.
4. Restarts CI and waits for the rollout.
5. Sets the secret on every matched hook and checks that GitHub now reports a secret.
6. Pings every hook and prints the status of the latest ping delivery. Expect 200.

It asks before each step that changes anything, and never prints or saves the secret. Between steps 4 and 5 GitHub's
deliveries are unsigned and get 401: expected, and covered by polling.

Use `--namespace` for a CI that isn't in `default`.

If the ping check fails, look for `github_webhook_rejected` on CI's events page and at the hook's
*Recent Deliveries* tab on GitHub (*Settings → Webhooks*).

## Rollback

```bash
python3 devbin/refresh-ci-webhook-secret.py --unset
```

This clears the secret on every matched hook first, then deletes the k8s secret, restarts CI and pings each hook.
Expect 200. CI is then back to accepting unsigned deliveries.

| Situation | Action |
|---|---|
| CI rejects valid deliveries | `--unset` (no code change needed) |
| Some other bug in the verification change | Revert the change **and** run `--unset`: the old code rejects every signed delivery |
| The script failed part-way | Rerun it (it's safe to rerun), or run `--unset` |
