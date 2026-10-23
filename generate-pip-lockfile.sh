#!/bin/bash
# Some service dependencies are linux-specific and the lockfile
# would differ when generated on MacOS, so we generate the lockfile
# on a linux image.
#
# --exclude-newer skips releases from the last 7 days, giving time for a
# compromised release to be noticed and yanked before we pick it up.

set -ex

package=$1
shift
if ! uv pip compile \
    --python-version 3.12 \
    --python-platform linux \
    --generate-hashes \
    --exclude-newer "7 days" \
    --upgrade $package/requirements.txt --output-file=$package/pinned-requirements.txt \
    "$@"; then
    set +x
    echo >&2
    echo "Releases from the last 7 days are excluded. If a requirement needs a newer release," >&2
    echo "exempt it with: make generate-pip-lockfiles COOLDOWN_EXEMPT=\"<package> ...\"" >&2
    echo "See dev-docs/pip-dependencies.md#release-cooldown" >&2
    exit 1
fi
