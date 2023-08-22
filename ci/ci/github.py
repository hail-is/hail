import asyncio
import concurrent.futures
import datetime
import json
import logging
import os
import random
import secrets
from enum import Enum
from shlex import quote as shq
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import aiohttp
import gidgethub
import prometheus_client as pc  # type: ignore
import zulip

from gear import Database, UserData
from hailtop.batch_client.aioclient import Batch, BatchClient
from hailtop.config import get_deploy_config
from hailtop.utils import RETRY_FUNCTION_SCRIPT, check_shell, check_shell_output

from .build import BuildConfiguration, Code
from .constants import AUTHORIZED_USERS, COMPILER_TEAM, GITHUB_CLONE_URL, GITHUB_STATUS_CONTEXT, SERVICES_TEAM
from .environment import DEPLOY_STEPS
from .globals import is_test_deployment
from .utils import add_deployed_services

repos_lock = asyncio.Lock()

log = logging.getLogger('ci')

deploy_config = get_deploy_config()

CALLBACK_URL = deploy_config.url('ci', '/api/v1alpha/batch_callback')

zulip_client: Optional[zulip.Client] = None
if os.path.exists("/zulip-config/.zuliprc"):
    zulip_client = zulip.Client(config_file="/zulip-config/.zuliprc")

TRACKED_PRS = pc.Gauge('ci_tracked_prs', 'PRs currently being monitored by CI', ['build_state', 'review_state'])

MAX_CONCURRENT_PR_BATCHES = 3


class GithubStatus(Enum):
    SUCCESS = 'success'
    PENDING = 'pending'
    FAILURE = 'failure'


def select_random_teammate(team):
    return random.choice([user for user in AUTHORIZED_USERS if team in user.teams])


async def sha_already_alerted(db: Database, sha: str) -> bool:
    record = await db.select_and_fetchone(
        '''
SELECT sha
FROM alerted_failed_shas
WHERE sha = %s
''',
        (sha,),
    )
    return record is not None


async def send_zulip_deploy_failure_message(message: str, db: Database, sha: Optional[str]):
    if zulip_client is None:
        log.info('Zulip integration is not enabled. No config file found')
        return

    if sha is not None:
        alerted = await sha_already_alerted(db, sha)
        if alerted:
            log.info(f'Already alerted failure to Zulip for sha {sha}')
            return

    request = {
        'type': 'stream',
        'to': 'team',
        'topic': 'CI Deploy Failure',
        'content': message,
    }
    result = zulip_client.send_message(request)
    log.info(result)

    if sha is not None:
        await db.execute_insertone(
            '''
INSERT INTO alerted_failed_shas (sha) VALUES (%s)
''',
            (sha,),
        )


class Repo:
    def __init__(self, owner, name):
        assert isinstance(owner, str)
        assert isinstance(name, str)
        self.owner = owner
        self.name = name
        self.url = f'{GITHUB_CLONE_URL}{owner}/{name}.git'

    def __eq__(self, other):
        return self.owner == other.owner and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.owner, self.name))

    def __str__(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_short_str(s):
        pieces = s.split("/")
        assert len(pieces) == 2, f'{pieces} {s}'
        return Repo(pieces[0], pieces[1])

    def short_str(self):
        return f'{self.owner}/{self.name}'

    @staticmethod
    def from_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'owner' in d, d
        assert 'name' in d, d
        return Repo(d['owner'], d['name'])

    def to_dict(self):
        return {'owner': self.owner, 'name': self.name}

    @staticmethod
    def from_gh_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'owner' in d, d
        assert 'login' in d['owner'], d
        assert 'name' in d, d
        return Repo(d['owner']['login'], d['name'])


class FQBranch:
    def __init__(self, repo, name):
        assert isinstance(repo, Repo)
        assert isinstance(name, str)
        self.repo = repo
        self.name = name

    def __eq__(self, other):
        return self.repo == other.repo and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.repo, self.name))

    def __str__(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_short_str(s):
        pieces = s.split(":")
        assert len(pieces) == 2, f'{pieces} {s}'
        return FQBranch(Repo.from_short_str(pieces[0]), pieces[1])

    def short_str(self) -> str:
        return f'{self.repo.short_str()}:{self.name}'

    @staticmethod
    def from_gh_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'repo' in d, d
        assert 'ref' in d, d
        return FQBranch(Repo.from_gh_json(d['repo']), d['ref'])

    @staticmethod
    def from_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'repo' in d, d
        assert 'name' in d, d
        return FQBranch(Repo.from_json(d['repo']), d['name'])

    def to_dict(self):
        return {'repo': self.repo.to_dict(), 'name': self.name}


# record the context for a merge failure
class MergeFailureBatch:
    def __init__(self, exception: BaseException, attributes: Dict[str, str]):
        self.exception = exception
        self.attributes = attributes


ASSIGN_SERVICES = '#assign services'
ASSIGN_COMPILER = '#assign compiler'

HIGH_PRIORITY = 'prio:high'
STACKED_PR = 'stacked PR'
WIP = 'WIP'

DO_NOT_MERGE = {STACKED_PR, WIP}


def clone_or_fetch_script(repo):
    return f"""
{ RETRY_FUNCTION_SCRIPT }

function clone() {{
    rm -rf ./{{*,.*}}
    git clone { shq(repo) } ./
}}

if [ ! -d .git ]; then
  time retry clone

  git config user.email ci@hail.is
  git config user.name ci
else
  git reset --hard
  time retry git fetch -q origin
fi
"""


class PR(Code):
    def __init__(
        self, number, title, body, source_branch, source_sha, target_branch, author, assignees, reviewers, labels
    ):
        self.number: int = number
        self.title: str = title
        self.body: str = body
        self.source_branch: FQBranch = source_branch
        self.source_sha: str = source_sha
        self.target_branch: 'WatchedBranch' = target_branch
        self.author: str = author
        self.assignees: Set[str] = assignees
        self.reviewers: Set[str] = reviewers
        self.labels: Set[str] = labels

        # pending, changes_requested, approve
        self.review_state: Optional[str] = None

        self.sha: Optional[str] = None
        self.batch: Union[Batch, MergeFailureBatch, None] = None
        self.source_sha_failed: Optional[bool] = None

        # 'error', 'success', 'failure', None
        self.build_state: Optional[str] = None

        self.intended_github_status: GithubStatus = self.github_status_from_build_state()
        self.last_known_github_status: Dict[str, GithubStatus] = {}

        # don't need to set github_changed because we are refreshing github
        self.target_branch.batch_changed = True
        self.target_branch.state_changed = True

    def set_build_state(self, build_state):
        log.info(f'{self.short_str()}: Build state changing from {self.build_state} => {build_state}')
        if build_state != self.build_state:
            self.decrement_pr_metric()
            self.build_state = build_state
            self.increment_pr_metric()

            intended_github_status = self.github_status_from_build_state()
            if intended_github_status != self.intended_github_status:
                self.intended_github_status = intended_github_status
                self.target_branch.state_changed = True

    def set_review_state(self, review_state):
        self.decrement_pr_metric()
        self.review_state = review_state
        self.increment_pr_metric()

    def decrement_pr_metric(self):
        TRACKED_PRS.labels(build_state=self.build_state, review_state=self.review_state).dec()

    def increment_pr_metric(self):
        TRACKED_PRS.labels(build_state=self.build_state, review_state=self.review_state).inc()

    async def authorized(self, db: Database):
        if self.author in {user.gh_username for user in AUTHORIZED_USERS}:
            return True

        row = await db.execute_and_fetchone('SELECT * from authorized_shas WHERE sha = %s;', self.source_sha)
        return row is not None

    def build_succeeding_on_all_platforms(self):
        return all(gh_status == GithubStatus.SUCCESS for gh_status in self.last_known_github_status.values())

    def build_failed_on_at_least_one_platform(self):
        return any(gh_status == GithubStatus.FAILURE for gh_status in self.last_known_github_status.values())

    def merge_priority(self):
        # passed > unknown > failed
        if self.build_succeeding_on_all_platforms():
            source_sha_failed_prio = 2
        elif self.build_failed_on_at_least_one_platform():
            source_sha_failed_prio = 0
        else:
            source_sha_failed_prio = 1

        return (
            HIGH_PRIORITY in self.labels,
            all(label not in DO_NOT_MERGE for label in self.labels),
            source_sha_failed_prio,
            # oldest first
            -self.number,
        )

    def short_str(self):
        return f'pr-{self.number}'

    def update_from_gh_json(self, gh_json):
        assert self.number == gh_json['number']
        self.title = gh_json['title']
        self.body = gh_json['body']
        self.author = gh_json['user']['login']
        self.assignees = {user['login'] for user in gh_json['assignees']}
        self.reviewers = {user['login'] for user in gh_json['requested_reviewers']}

        new_labels = {label['name'] for label in gh_json['labels']}
        if new_labels != self.labels:
            self.labels = new_labels
            self.target_branch.state_changed = True

        head = gh_json['head']
        new_source_sha = head['sha']
        if self.source_sha != new_source_sha:
            log.info(f'{self.short_str()} source sha changed: {self.source_sha} => {new_source_sha}')
            self.source_sha = new_source_sha
            self.sha = None
            self.batch = None
            self.source_sha_failed = None
            self.set_build_state(None)
            self.target_branch.batch_changed = True
            self.target_branch.state_changed = True

        self.source_branch = FQBranch.from_gh_json(head)

    @staticmethod
    def from_gh_json(gh_json, target_branch):
        head = gh_json['head']
        pr = PR(
            gh_json['number'],
            gh_json['title'],
            gh_json['body'],
            FQBranch.from_gh_json(head),
            head['sha'],
            target_branch,
            gh_json['user']['login'],
            {user['login'] for user in gh_json['assignees']},
            {user['login'] for user in gh_json['requested_reviewers']},
            {label['name'] for label in gh_json['labels']},
        )
        pr.increment_pr_metric()
        return pr

    def repo_dir(self):
        return self.target_branch.repo_dir()

    def config(self):
        assert self.sha is not None
        source_repo = self.source_branch.repo
        target_repo = self.target_branch.branch.repo
        return {
            'checkout_script': self.checkout_script(),
            'number': self.number,
            'source_repo': source_repo.short_str(),
            'source_repo_url': source_repo.url,
            'source_sha': self.source_sha,
            'target_repo': target_repo.short_str(),
            'target_repo_url': target_repo.url,
            'target_sha': self.target_branch.sha,
            'sha': self.sha,
        }

    def github_status_from_build_state(self) -> GithubStatus:
        if self.build_state in ('failure', 'error'):
            return GithubStatus.FAILURE
        if (
            self.build_state == 'success'
            and self.batch
            and self.batch.attributes['target_sha'] == self.target_branch.sha
        ):
            return GithubStatus.SUCCESS
        return GithubStatus.PENDING

    async def post_github_status(self, gh_client, gh_status: GithubStatus):
        assert self.source_sha is not None

        log.info(f'{self.short_str()}: notify github state: {gh_status}')
        if self.batch is None or isinstance(self.batch, MergeFailureBatch):
            target_url = deploy_config.external_url(
                'ci', f'/watched_branches/{self.target_branch.index}/pr/{self.number}'
            )
        else:
            assert self.batch.id is not None
            target_url = deploy_config.external_url('ci', f'/batches/{self.batch.id}')
        data = {
            'state': gh_status.value,
            'target_url': target_url,
            # FIXME improve
            'description': gh_status.value,
            'context': GITHUB_STATUS_CONTEXT,
        }
        try:
            await gh_client.post(
                f'/repos/{self.target_branch.branch.repo.short_str()}/statuses/{self.source_sha}', data=data
            )
        except gidgethub.HTTPException:
            log.exception(f'{self.short_str()}: notify github of build state failed due to exception: {data}')
        except aiohttp.client_exceptions.ClientResponseError:
            log.exception(f'{self.short_str()}: Unexpected exception in post to github: {data}')

    async def assign_gh_reviewer_if_requested(self, gh_client):
        if len(self.assignees) == 0 and len(self.reviewers) == 0 and self.body is not None:
            assignees = set()
            if ASSIGN_SERVICES in self.body:
                assignees.add(select_random_teammate(SERVICES_TEAM).gh_username)
            if ASSIGN_COMPILER in self.body:
                assignees.add(select_random_teammate(COMPILER_TEAM).gh_username)
            data = {'assignees': list(assignees)}
            try:
                await gh_client.post(
                    f'/repos/{self.target_branch.branch.repo.short_str()}/issues/{self.number}/assignees', data=data
                )
            except gidgethub.HTTPException:
                log.exception(f'{self.short_str()}: post assignees to github failed due to exception: {data}')
            except aiohttp.client_exceptions.ClientResponseError:
                log.exception(f'{self.short_str()}: Unexpected exception in post to github: {data}')

    async def _update_github(self, gh):
        await self._update_last_known_github_status(gh)
        await self._update_github_review_state(gh)

    @staticmethod
    def _hail_github_status_from_statuses(statuses_json) -> Dict[str, GithubStatus]:
        statuses = statuses_json["statuses"]
        hail_statuses = {}
        for s in statuses:
            context = s['context']
            if context == 'ci-test' or context.startswith('hail-ci'):
                if context in hail_statuses:
                    raise ValueError(
                        f'github sent multiple status summaries for context {context}: {s}\n\n{statuses_json}'
                    )
                hail_statuses[context] = GithubStatus(s['state'])
        return hail_statuses

    async def _update_last_known_github_status(self, gh):
        if self.source_sha:
            source_sha_json = await gh.getitem(
                f'/repos/{self.target_branch.branch.repo.short_str()}/commits/{self.source_sha}/status'
            )
            last_known_github_status = PR._hail_github_status_from_statuses(source_sha_json)
            if last_known_github_status != self.last_known_github_status:
                log.info(
                    f'{self.short_str()}: new github statuses: {self.last_known_github_status} => {last_known_github_status}'
                )
                self.last_known_github_status = last_known_github_status
                self.target_branch.state_changed = True

    async def _update_github_review_state(self, gh):
        latest_state_by_login = {}
        async for review in gh.getiter(
            f'/repos/{self.target_branch.branch.repo.short_str()}/pulls/{self.number}/reviews'
        ):
            login = review['user']['login']
            state = review['state']
            # reviews is chronological, so later ones are newer statuses
            if state != 'COMMENTED':
                latest_state_by_login[login] = state

        review_state = 'pending'
        for login, state in latest_state_by_login.items():
            if state == 'CHANGES_REQUESTED':
                review_state = 'changes_requested'
                break
            if state == 'APPROVED':
                review_state = 'approved'
            else:
                assert state in ('DISMISSED', 'COMMENTED', 'PENDING'), state

        if review_state != self.review_state:
            self.set_review_state(review_state)
            self.target_branch.state_changed = True

    async def _start_build(self, db: Database, batch_client: BatchClient):
        assert await self.authorized(db)

        # clear current batch
        self.batch = None
        self.set_build_state(None)

        batch = None
        try:
            log.info(f'merging for {self.number}')
            repo_dir = self.repo_dir()
            await check_shell(
                f'''
set -ex
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
'''
            )

            sha_out, _ = await check_shell_output(f'git -C {shq(repo_dir)} rev-parse HEAD')
            self.sha = sha_out.decode('utf-8').strip()

            with open(f'{repo_dir}/build.yaml', 'r', encoding='utf-8') as f:
                config = BuildConfiguration(self, f.read(), scope='test')
                namespace = config.namespace()
                services = config.deployed_services()
            with open(f'{repo_dir}/ci/test/resources/build.yaml', 'r', encoding='utf-8') as f:
                test_services = BuildConfiguration(self, f.read(), scope='test').deployed_services()

            services.extend(test_services)
            tomorrow = datetime.datetime.utcnow() + datetime.timedelta(days=1)
            assert namespace is not None
            await add_deployed_services(db, namespace, services, tomorrow)

            log.info(f'creating test batch for {self.number}')
            batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'test': '1',
                    'source_branch': self.source_branch.short_str(),
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'namespace': namespace,
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha,
                },
                callback=CALLBACK_URL,
            )
            config.build(batch, self, scope='test')
            await batch.submit()
            self.batch = batch
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            # FIXME save merge failure output for UI
            assert self.target_branch.sha is not None
            self.batch = MergeFailureBatch(
                e,
                attributes={
                    'test': '1',
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha,
                },
            )
            self.set_build_state('error')
            self.source_sha_failed = True
            self.target_branch.state_changed = True
        finally:
            if batch and not self.batch:
                log.info(f'cancelling partial test batch {batch.id}')
                await batch.cancel()

    @staticmethod
    async def is_invalidated_batch(batch, db: Database):
        assert batch is not None
        row = await db.execute_and_fetchone('SELECT * from invalidated_batches WHERE batch_id = %s;', batch.id)
        return row is not None

    async def _update_batch(self, batch_client, db: Database):
        # find the latest non-cancelled batch for source
        batches = batch_client.list_batches(
            f'test=1 '
            f'target_branch={self.target_branch.branch.short_str()} '
            f'source_sha={self.source_sha} '
            f'user:ci'
        )
        min_batch = None
        min_batch_status = None
        async for b in batches:
            if await self.is_invalidated_batch(b, db):
                continue
            try:
                s = await b.status()
            except Exception:
                log.exception(f'failed to get the status for batch {b.id}')
                raise
            if s['state'] != 'cancelled':
                if min_batch is None or b.id > min_batch.id:
                    min_batch = b
                    min_batch_status = s
        self.batch = min_batch
        self.source_sha_failed = None

        if min_batch_status is None:
            self.set_build_state(None)
        elif min_batch_status['complete']:
            if min_batch_status['state'] == 'success':
                self.set_build_state('success')
                self.source_sha_failed = False
            else:
                self.set_build_state('failure')
                self.source_sha_failed = True
            self.target_branch.state_changed = True

    async def _heal(self, batch_client, db: Database, on_deck, gh):
        # can't merge target if we don't know what it is
        if self.target_branch.sha is None:
            return

        if self.source_sha:
            last_posted_status = self.last_known_github_status.get(GITHUB_STATUS_CONTEXT)
            if self.intended_github_status != last_posted_status:
                log.info(f'Intended github status for {self.short_str()} is: {self.intended_github_status}')
                log.info(f'Last known github status for {self.short_str()} is: {last_posted_status}')
                await self.post_github_status(gh, self.intended_github_status)
                self.last_known_github_status[GITHUB_STATUS_CONTEXT] = self.intended_github_status

        if not await self.authorized(db):
            return

        if not self.batch or (on_deck and self.batch.attributes['target_sha'] != self.target_branch.sha):
            if on_deck or self.target_branch.n_running_batches < MAX_CONCURRENT_PR_BATCHES:
                self.target_branch.n_running_batches += 1
                async with repos_lock:
                    await self._start_build(db, batch_client)

    def is_up_to_date(self):
        return self.batch is not None and self.target_branch.sha == self.batch.attributes['target_sha']

    def is_mergeable(self) -> bool:
        if self.last_known_github_status.get(GITHUB_STATUS_CONTEXT) == GithubStatus.SUCCESS:
            assert self.build_state == 'success', (
                self.last_known_github_status[GITHUB_STATUS_CONTEXT],
                self.build_state,
            )
        return (
            self.review_state == 'approved'
            and len(self.last_known_github_status) > 0
            and all(status == GithubStatus.SUCCESS for status in self.last_known_github_status.values())
            and self.is_up_to_date()
            and all(label not in DO_NOT_MERGE for label in self.labels)
        )

    async def merge(self, gh):
        try:
            await gh.put(
                f'/repos/{self.target_branch.branch.repo.short_str()}/pulls/{self.number}/merge',
                data={'merge_method': 'squash', 'sha': self.source_sha},
            )
            return True
        except (gidgethub.HTTPException, aiohttp.client_exceptions.ClientResponseError):
            log.info(f'merge {self.target_branch.branch.short_str()} {self.number} failed', exc_info=True)
        return False

    def checkout_script(self):
        assert self.target_branch.sha
        return f'''
{clone_or_fetch_script(self.target_branch.branch.repo.url)}

git remote add {shq(self.source_branch.repo.short_str())} {shq(self.source_branch.repo.url)} || true

time retry git fetch -q {shq(self.source_branch.repo.short_str())}
git checkout {shq(self.target_branch.sha)}
git merge {shq(self.source_sha)} -m 'merge PR'
'''


class WatchedBranch(Code):
    def __init__(self, index, branch, deployable, mergeable, developers):
        self.index: int = index
        self.branch: FQBranch = branch
        self.deployable: bool = deployable
        self.mergeable: bool = mergeable
        self.developers: List[dict] = developers

        self.prs: Dict[int, PR] = {}
        self.sha: Optional[str] = None

        self.deploy_batch: Union[Batch, MergeFailureBatch, None] = None
        # success, failure, pending
        self._deploy_state: Optional[str] = None

        self.updating: bool = False
        self.github_changed: bool = True
        self.batch_changed: bool = True
        self.state_changed: bool = True

        self.n_running_batches: int = 0

        self.merge_candidate: Optional[PR] = None

    @property
    def deploy_state(self):
        return self._deploy_state

    @deploy_state.setter
    def deploy_state(self, new_state):
        self._deploy_state = new_state

    def short_str(self):
        return f'br-{self.branch.repo.owner}-{self.branch.repo.name}-{self.branch.name}'

    def repo_dir(self):
        return f'repos/{self.branch.repo.short_str()}'

    def config(self):
        assert self.sha is not None
        return {
            'checkout_script': self.checkout_script(),
            'branch': self.branch.name,
            'repo': self.branch.repo.short_str(),
            'repo_url': self.branch.repo.url,
            'sha': self.sha,
            'developers': self.developers,
        }

    async def notify_github_changed(self, app):
        self.github_changed = True
        await self._update(app)

    async def notify_batch_changed(self, app):
        self.batch_changed = True
        await self._update(app)

    async def update(self, app):
        # update everything
        self.github_changed = True
        self.batch_changed = True
        self.state_changed = True
        await self._update(app)

    async def _update(self, app):
        if self.updating:
            log.info(f'already updating {self.short_str()}')
            return

        try:
            log.info(f'start update {self.short_str()}')
            self.updating = True
            gh = app['github_client']
            batch_client = app['batch_client']
            db: Database = app['db']

            while self.github_changed or self.batch_changed or self.state_changed:
                if self.github_changed:
                    self.github_changed = False
                    await self._update_github(gh)

                if self.batch_changed:
                    self.batch_changed = False
                    await self._update_batch(batch_client, db)

                if self.state_changed:
                    self.state_changed = False
                    await self._heal(app, batch_client, gh)
                    if (
                        (self.deploy_batch is None or self.deploy_state is not None)
                        and not app['frozen_merge_deploy']
                        and self.mergeable
                    ):
                        await self.try_to_merge(gh)
        finally:
            log.info(f'update done {self.short_str()}')
            self.updating = False

    async def try_to_merge(self, gh):
        assert self.mergeable
        for pr in self.prs.values():
            if pr.is_mergeable():
                if await pr.merge(gh):
                    self.github_changed = True
                    self.sha = None
                    self.state_changed = True
                    self.merge_candidate = None
                    return

    async def _update_github(self, gh):
        log.info(f'update github {self.short_str()}')

        repo_ss = self.branch.repo.short_str()

        branch_gh_json = await gh.getitem(f'/repos/{repo_ss}/git/refs/heads/{self.branch.name}')
        new_sha = branch_gh_json['object']['sha']
        if new_sha != self.sha:
            log.info(f'{self.branch.short_str()} sha changed: {self.sha} => {new_sha}')
            self.sha = new_sha
            self.state_changed = True

        new_prs: Dict[int, PR] = {}
        async for gh_json_pr in gh.getiter(f'/repos/{repo_ss}/pulls?state=open&base={self.branch.name}'):
            number = gh_json_pr['number']
            if number in self.prs:
                pr = self.prs[number]
                pr.update_from_gh_json(gh_json_pr)
            else:
                pr = PR.from_gh_json(gh_json_pr, self)
            new_prs[number] = pr
        for number, pr in self.prs.items():
            if number not in new_prs:
                pr.decrement_pr_metric()
        self.prs = new_prs

        for pr in new_prs.values():
            await pr.assign_gh_reviewer_if_requested(gh)

        for pr in new_prs.values():
            await pr._update_github(gh)

    async def _update_deploy(self, batch_client, db: Database):
        assert self.deployable

        if self.deploy_state:
            assert self.deploy_batch
            return

        if self.deploy_batch is None:
            running_deploy_batches = batch_client.list_batches(
                f'!complete ' f'deploy=1 ' f'target_branch={self.branch.short_str()} ' f'user:ci'
            )
            running_deploy_batches = [b async for b in running_deploy_batches]
            if running_deploy_batches:
                self.deploy_batch = max(running_deploy_batches, key=lambda b: b.id)
            else:
                deploy_batches = batch_client.list_batches(
                    f'deploy=1 ' f'target_branch={self.branch.short_str()} ' f'sha={self.sha} ' f'user:ci'
                )
                deploy_batches = [b async for b in deploy_batches]
                if deploy_batches:
                    self.deploy_batch = max(deploy_batches, key=lambda b: b.id)

        if self.deploy_batch:
            assert isinstance(self.deploy_batch, Batch)
            try:
                status = await self.deploy_batch.status()
            except aiohttp.client_exceptions.ClientResponseError as exc:
                log.exception(
                    f'Could not update deploy_batch status due to exception {exc}, setting deploy_batch to None'
                )
                self.deploy_batch = None
                return
            if status['complete']:
                if status['state'] == 'success':
                    self.deploy_state = 'success'
                else:
                    self.deploy_state = 'failure'

                if not is_test_deployment and self.deploy_state == 'failure':
                    url = deploy_config.external_url('ci', f'/batches/{self.deploy_batch.id}')
                    deploy_failure_message = f'''
state: {self.deploy_state}
branch: {self.branch.short_str()}
sha: {self.sha}
url: {url}
'''
                    await send_zulip_deploy_failure_message(deploy_failure_message, db, self.sha)
                self.state_changed = True

    async def _heal_deploy(self, app, batch_client):
        assert self.deployable

        if not self.sha:
            return

        if not app['frozen_merge_deploy'] and (
            self.deploy_batch is None or (self.deploy_state and self.deploy_batch.attributes['sha'] != self.sha)
        ):
            async with repos_lock:
                await self._start_deploy(app['db'], batch_client)

    async def _update_batch(self, batch_client: BatchClient, db: Database):
        log.info(f'update batch {self.short_str()}')

        if self.deployable:
            await self._update_deploy(batch_client, db)

        for pr in self.prs.values():
            await pr._update_batch(batch_client, db)

    async def _heal(self, app, batch_client, gh):
        log.info(f'heal {self.short_str()}')
        db: Database = app['db']

        if self.deployable:
            await self._heal_deploy(app, batch_client)

        merge_candidate = None
        merge_candidate_pri = None
        for pr in self.prs.values():
            # merge candidate if up-to-date build passing, or
            # pending but haven't failed
            if pr.review_state == 'approved' and not pr.build_failed_on_at_least_one_platform():
                pri = pr.merge_priority()
                is_authorized = await pr.authorized(db)
                if is_authorized and (not merge_candidate or pri > merge_candidate_pri):
                    merge_candidate = pr
                    merge_candidate_pri = pri

        self.merge_candidate = merge_candidate

        if merge_candidate:
            log.info(f'merge candidate {merge_candidate.number}')

        self.n_running_batches = sum(1 for pr in self.prs.values() if pr.batch and not pr.build_state)

        for pr in self.prs.values():
            await pr._heal(batch_client, db, pr == merge_candidate, gh)

        # cancel orphan builds
        running_batches = batch_client.list_batches(
            f'!complete ' f'!open ' f'test=1 ' f'target_branch={self.branch.short_str()} ' f'user:ci'
        )
        seen_batch_ids = set(pr.batch.id for pr in self.prs.values() if pr.batch and isinstance(pr.batch, Batch))
        async for batch in running_batches:
            if batch.id not in seen_batch_ids:
                attrs = batch.attributes
                log.info(f'cancel batch {batch.id} for {attrs["pr"]} {attrs["source_sha"]} => {attrs["target_sha"]}')
                await batch.cancel()

    async def _start_deploy(self, db: Database, batch_client: BatchClient):
        # not deploying
        assert not self.deploy_batch or self.deploy_state

        self.deploy_batch = None
        self.deploy_state = None

        deploy_batch = None
        assert self.sha is not None
        try:
            repo_dir = self.repo_dir()
            await check_shell(
                f'''
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
'''
            )
            with open(f'{repo_dir}/build.yaml', 'r', encoding='utf-8') as f:
                config = BuildConfiguration(self, f.read(), requested_step_names=DEPLOY_STEPS, scope='deploy')
                namespace = config.namespace()
                services = config.deployed_services()
            with open(f'{repo_dir}/ci/test/resources/build.yaml', 'r', encoding='utf-8') as f:
                test_services = BuildConfiguration(self, f.read(), scope='deploy').deployed_services()

            services.extend(test_services)
            assert namespace is not None
            await add_deployed_services(db, namespace, services, None)

            log.info(f'creating deploy batch for {self.branch.short_str()}')
            deploy_batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'deploy': '1',
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha,
                },
                callback=CALLBACK_URL,
            )
            try:
                config.build(deploy_batch, self, scope='deploy')
            except Exception as e:
                deploy_failure_message = f'''
branch: {self.branch.short_str()}
sha: {self.sha}
Deploy config failed to build with exception:
```python
{e}
```
'''
                await send_zulip_deploy_failure_message(deploy_failure_message, db, self.sha)
                raise
            await deploy_batch.submit()
            self.deploy_batch = deploy_batch
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            log.exception('could not start deploy')
            self.deploy_batch = MergeFailureBatch(
                e, attributes={'deploy': '1', 'target_branch': self.branch.short_str(), 'sha': self.sha}
            )
            self.deploy_state = 'checkout_failure'
        finally:
            if deploy_batch and not self.deploy_batch:
                log.info(f'deleting partially deployed batch {deploy_batch.id}')
                await deploy_batch.delete()

    def checkout_script(self) -> str:
        assert self.sha
        return f'''
{clone_or_fetch_script(self.branch.repo.url)}

git checkout {shq(self.sha)}
'''


class UnwatchedBranch(Code):
    def __init__(
        self,
        branch: FQBranch,
        sha: str,
        userdata: UserData,
        developers: List[UserData],
        *,
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        self.branch = branch
        self.user: str = userdata['username']
        self.namespace: str = userdata['namespace_name']
        self.developers = developers
        self.sha = sha
        self.extra_config = extra_config

        self.deploy_batch: Optional[Batch] = None

    def short_str(self) -> str:
        return f'br-{self.branch.repo.owner}-{self.branch.repo.name}-{self.branch.name}'

    def repo_dir(self) -> str:
        return f'repos/{self.branch.repo.short_str()}'

    def config(self) -> Dict[str, str]:
        config = {
            'checkout_script': self.checkout_script(),
            'branch': self.branch.name,
            'repo': self.branch.repo.short_str(),
            'repo_url': self.branch.repo.url,
            'sha': self.sha,
            'user': self.user,
            'developers': self.developers,
        }
        if self.extra_config is not None:
            config.update(self.extra_config)
        return config

    async def deploy(
        self, db: Database, batch_client: BatchClient, steps: Sequence[str], excluded_steps: Sequence[str] = ()
    ):
        assert not self.deploy_batch

        deploy_batch = None
        try:
            repo_dir = self.repo_dir()
            await check_shell(
                f'''
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
'''
            )
            log.info(f'User {self.user} requested these steps for dev deploy: {steps}')
            with open(f'{repo_dir}/build.yaml', 'r', encoding='utf-8') as f:
                config = BuildConfiguration(
                    self, f.read(), scope='dev', requested_step_names=steps, excluded_step_names=excluded_steps
                )
                namespace = config.namespace()
                services = config.deployed_services()
            with open(f'{repo_dir}/ci/test/resources/build.yaml', 'r', encoding='utf-8') as f:
                test_services = BuildConfiguration(self, f.read(), scope='dev').deployed_services()
            if namespace is not None:
                services.extend(test_services)
                await add_deployed_services(db, namespace, services, None)

            log.info(f'creating dev deploy batch for {self.branch.short_str()} and user {self.user}')

            deploy_batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha,
                    'user': self.user,
                    'dev_deploy': '1',
                }
            )
            config.build(deploy_batch, self, scope='dev')
            await deploy_batch.submit()
            self.deploy_batch = deploy_batch
            return deploy_batch.id
        finally:
            if deploy_batch and not self.deploy_batch and isinstance(deploy_batch, Batch):
                log.info(f'deleting partially created deploy batch {deploy_batch.id}')
                await deploy_batch.delete()

    def checkout_script(self) -> str:
        return f'''
{clone_or_fetch_script(self.branch.repo.url)}

git checkout {shq(self.sha)}
'''
