#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_FILE="$REPO_ROOT/docker/third-party/images.txt"
DOCKERFILE="$REPO_ROOT/docker/hail-ubuntu/Dockerfile"

echo "Querying Docker Hub for latest ubuntu noble tag..."
LATEST_TAG=$(python3 - <<'EOF'
import urllib.request, json, re, sys

url = 'https://hub.docker.com/v2/repositories/library/ubuntu/tags?page_size=100&name=noble-2'
with urllib.request.urlopen(url) as r:
    data = json.loads(r.read())

pattern = re.compile(r'^noble-(\d{8})$')
tags = [r['name'] for r in data['results'] if pattern.match(r['name'])]
if not tags:
    print('No noble-YYYYMMDD tags found', file=sys.stderr)
    sys.exit(1)
tags.sort()
print(tags[-1])
EOF
)

CURRENT_TAG=$(grep '^ubuntu:noble-' "$IMAGES_FILE" | sed 's/ubuntu://')

if [ "$CURRENT_TAG" = "$LATEST_TAG" ]; then
    echo "Already up to date: ubuntu:${LATEST_TAG}"
    exit 2
fi

echo "Updating ubuntu:${CURRENT_TAG} -> ubuntu:${LATEST_TAG}"

python3 - "$IMAGES_FILE" "$CURRENT_TAG" "$LATEST_TAG" <<'EOF'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
updated = text.replace(f'ubuntu:{old}', f'ubuntu:{new}')
assert updated != text, f'Pattern ubuntu:{old} not found in {path}'
open(path, 'w').write(updated)
EOF

python3 - "$DOCKERFILE" "$CURRENT_TAG" "$LATEST_TAG" <<'EOF'
import sys
path, old, new = sys.argv[1:]
text = open(path).read()
updated = text.replace(old, new)
assert updated != text, f'Pattern {old} not found in {path}'
open(path, 'w').write(updated)
EOF

echo "Updated docker/third-party/images.txt and docker/hail-ubuntu/Dockerfile"
