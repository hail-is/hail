import json
import os

import uuid
from batch.api import API
from batch.client import BatchClient

from .ci_logging import log
from .git_state import FQRef

INSTANCE_ID = uuid.uuid4().hex

SELF_HOSTNAME = os.environ.get('SELF_HOSTNAME',
                               'http://set_the_SELF_HOSTNAME/')
BATCH_SERVER_URL = os.environ.get('BATCH_SERVER_URL',
                                  'http://set_the_BATCH_SERVER_URL/')
REFRESH_INTERVAL_IN_SECONDS = \
    int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 60))

CONTEXT = f'hail-ci-0-1'

try:
    WATCHED_TARGETS = [
        (FQRef.from_short_str(ref), deployable)
        for [ref, deployable] in json.loads(os.environ.get('WATCHED_TARGETS', '[]'))
    ]
except Exception as e:
    raise ValueError(
        'environment variable WATCHED_TARGETS should be a json array of arrays of refs and deployability as '
        f'strings and booleans e.g. [["hail-is/hail:master", true]], but was: `{os.environ.get("WATCHED_TARGETS", None)}`',
    ) from e
try:
    with open('pr-build-script', 'r') as f:
        PR_BUILD_SCRIPT = f.read()
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain a file called `pr-build-script' "
        "containing a string that is passed to `/bin/sh -c'") from e
try:
    with open('pr-deploy-script', 'r') as f:
        PR_DEPLOY_SCRIPT = f.read()
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain a file called `pr-deploy-script' "
        "containing a string that is passed to `/bin/sh -c'") from e
try:
    with open('oauth-token/oauth-token', 'r') as f:
        oauth_token = f.read().strip()
except FileNotFoundError as e:
    raise ValueError(
        "working directory must contain `oauth-token/oauth-token' "
        "containing a valid GitHub oauth token") from e

log.info(f'BATCH_SERVER_URL {BATCH_SERVER_URL}')
log.info(f'SELF_HOSTNAME {SELF_HOSTNAME}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'WATCHED_TARGETS {[(ref.short_str(), deployable) for (ref, deployable) in WATCHED_TARGETS]}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')
log.info(f'CONTEXT = {CONTEXT}')


batch_client = BatchClient(url=BATCH_SERVER_URL,
                           timeout=5,
                           cookies={'user': {'id': 0, 'email': 'ci@hail.is'}})
