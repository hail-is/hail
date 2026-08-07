---
disable-model-invocation: true
description: Prepare a Hail release — bump the patch version and curate the query and batch changelogs from commits since the last release tag
---

You are preparing the changelog portion of a quarterly Hail release. The deliverable is a version
bump plus curated, user-facing changelog entries in two files. A human reviews the result, so flag
any borderline noteworthiness calls in your final summary rather than silently dropping them.

## Steps

### 1. Bump the patch version

Two places, which must agree (skip if already done — check `git diff`):

- `hail/Makefile`: `HAIL_PATCH_VERSION := <n+1>`
- `hail/build.mill`: `val hailPatchVersion = "<n+1>"`

The previous version is the latest git tag (e.g. `0.2.138`): `git describe --tags $(git rev-list
--tags --max-count=1)`.

### 2. Generate raw changelog entries

Run the helper script from the `hail/` directory (it uses relative paths):

```
cd hail && scripts/update-changelog
```

It prepends a new version section with one line per commit to both changelogs, filtered by the
`[tag]` in each commit subject (query-ish tags → `python/hail/docs/change_log.md`, service-ish
tags → `python/hailtop/batch/docs/change_log.rst`). Skip this step if the raw entries are already
present.

### 3. Gather context on each commit

PRs are squash-merged, so **the commit body contains the full PR description** — do NOT fetch PR
metadata from GitHub for this. One local command gives everything:

```
git log --format='%h %s%n%b%n---' <prev-tag>..HEAD
```

Only reach for `gh pr view` when you need post-merge context: review comments, linked issues, or a
description edited after merge.

### 4. Curate

Keep entries that affect users; drop internal noise; reword keepers into each file's style.

**Noteworthy:**
- New features, new API surface (functions, parameters, CLI commands, endpoints)
- Bug fixes users could have hit, including performance regressions
- Performance improvements users will notice
- Deprecations and removals (including parameters becoming no-ops)
- File-format or on-disk layout changes — always state the compatibility direction explicitly
  (typically: new version reads old files, old versions cannot read new files)
- Major dependency changes: Python, Spark, Scala, Java versions
- Notable infrastructure changes users see in behavior or billing (e.g. worker machine family)

**Not noteworthy:**
- Internal refactors (compiler internals, no-sharing invariants, code organisation)
- Build/CI/test tooling changes
- Routine dependency bumps: pip/npm packages, Ubuntu base images, build tools
- Logging tweaks and docs-only nits
- Bugs both introduced and fixed since the previous tag — no released version ever had them

**Verify every bug-fix entry against the previous release.** Only bugs that existed in a released
version belong in the changelog. For each fix, find the commit that introduced the bug (the PR
body often names the culprit PR; otherwise use the diff, `git log -S`, or `git log --follow` on
the touched code) and check whether it shipped:

```
git merge-base --is-ancestor <introducing-commit> <prev-tag> && echo "shipped — keep" || echo "drop"
```

If the culprit can't be pinned down, keep the entry and flag it in the summary.

When a commit's noteworthiness or user-visible symptom is unclear from its message, read the PR
body (already in the commit body), the linked issue, or the diff. If still unsure, keep your best
judgment in the file and flag it in the summary.

### 5. Reword to match each file's style

Study the previous version's section in each file first; match it. Use American English spellings
(behavior, materialize, randomization — not behaviour, materialise, randomisation).

**Query changelog** (`hail/python/hail/docs/change_log.md`, Markdown):

```
## Version 0.2.NNN

Released YYYY-MM-DD

### New Features

- (hail#15445) Table and matrix table index files are now written as a single file ...

### Bug Fixes

- (hail#15374) Fix a bug where `sparse_split_multi` crashed on haploid genotypes ...

### Deprecations

- (hail#15589) The `stage_locally` parameter of write methods no longer has any effect ...
```

- Three `###` subsections: New Features, Bug Fixes, Deprecations (omit an empty subsection).
- Perf improvements go under New Features; perf *regression* fixes under Bug Fixes ("Fix a
  performance regression where ...").
- Bug fixes start "Fix a bug where ..." and describe the user-visible symptom, naming the Python
  API (`hl.foo`, backtick-quoted), not compiler internals.
- The script leaves an empty section skeleton and trailing whitespace on headings — clean both up.

**Batch changelog** (`hail/python/hailtop/batch/docs/change_log.rst`, RST):

```
**Version 0.2.NNN**

- (`#15200 <https://github.com/hail-is/hail/pull/15200>`__) All services now use ...
```

- Flat list, no subsections. Order: major changes first, then features, then fixes.
- Double backticks for code (RST): ``hailctl batch jobs``.
- Audience includes end users of hailtop.batch/hailctl AND operators of Hail deployments; admin
  UI button-level fixes are usually too minor unless they shipped broken in a prior release.
- Several PRs delivering one change (e.g. the same feature split per-service) may share one entry
  with all PR links.

### 6. Summarise for review

End with: what you kept per file, what you dropped and why (grouped, not exhaustive), and the
specific judgment calls the releaser should double-check.
