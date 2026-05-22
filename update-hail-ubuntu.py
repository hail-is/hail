#!/usr/bin/env python3
import re
import sys
import urllib.request
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent
IMAGES_FILE = REPO_ROOT / 'docker/third-party/images.txt'
DOCKERFILE = REPO_ROOT / 'docker/hail-ubuntu/Dockerfile'

url = 'https://hub.docker.com/v2/repositories/library/ubuntu/tags?page_size=100&name=noble-2'
print('Querying Docker Hub for latest ubuntu noble tag...')
with urllib.request.urlopen(url) as r: # nosec B310 - ignore bandit rule because this url is https
    data = json.loads(r.read())

pattern = re.compile(r'^noble-(\d{8})$')
tags = [result['name'] for result in data['results'] if pattern.match(result['name'])]
if not tags:
    print('ERROR: No noble-YYYYMMDD tags found', file=sys.stderr)
    sys.exit(1)
latest_tag = sorted(tags)[-1]

images_text = IMAGES_FILE.read_text()
m = re.search(r'^ubuntu:(noble-\d{8})$', images_text, re.MULTILINE)
if not m:
    print('ERROR: No ubuntu:noble-YYYYMMDD line found in images.txt', file=sys.stderr)
    sys.exit(1)
current_tag = m.group(1)

if current_tag == latest_tag:
    print(f'Already up to date: ubuntu:{latest_tag}')
    sys.exit(1)

print(f'Updating ubuntu:{current_tag} -> ubuntu:{latest_tag}')
IMAGES_FILE.write_text(images_text.replace(f'ubuntu:{current_tag}', f'ubuntu:{latest_tag}'))

dockerfile_text = DOCKERFILE.read_text()
DOCKERFILE.write_text(dockerfile_text.replace(current_tag, latest_tag))

print('Updated docker/third-party/images.txt and docker/hail-ubuntu/Dockerfile')
