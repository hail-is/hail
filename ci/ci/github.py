from typing import Dict, Optional
import secrets
from shlex import quote as shq
import json
import logging
import asyncio
import concurrent.futures
import aiohttp
import gidgethub
import zulip
import random
import prometheus_client as pc  # type: ignore

from hailtop.config import get_deploy_config
from hailtop.batch_client.aioclient import Batch
from hailtop.utils import check_shell, check_shell_output, RETRY_FUNCTION_SCRIPT
from .constants import GITHUB_CLONE_URL, AUTHORIZED_USERS, GITHUB_STATUS_CONTEXT, SERVICES_TEAM, COMPILER_TEAM
from .build import BuildConfiguration, Code
from .globals import is_test_deployment
from .environment import DEPLOY_STEPS


repos_lock = asyncio.Lock()

log = logging.getLogger('ci')

deploy_config = get_deploy_config()

CALLBACK_URL = deploy_config.url('ci', '/api/v1alpha/batch_callback')

zulip_client = zulip.Client(config_file="/zulip-config/.zuliprc")

TRACKED_PRS = pc.Gauge('ci_tracked_prs', 'PRs currently being monitored by CI', ['build_state', 'review_state'])


def select_random_teammate(team):
    return random.choice([user for user in AUTHORIZED_USERS if team in user.teams])


def send_zulip_deploy_failure_message(message):
    request = {
        'type': 'stream',
        'to': 'team',
        'topic': 'CI Deploy Failure',
        'content': message,
    }
    result = zulip_client.send_message(request)
    log.info(result)


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

    def short_str(self):
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
    def __init__(self, exception, attributes=None):
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
        self.number = number
        self.title = title
        self.body = body
        self.source_branch = source_branch
        self.source_sha = source_sha
        self.target_branch = target_branch
        self.author = author
        self.assignees = assignees
        self.reviewers = reviewers
        self.labels = labels

        # pending, changes_requested, approve
        self.review_state = None

        self.sha = None
        self.batch = None
        self.source_sha_failed = None

        # 'error', 'success', 'failure', None
        self.build_state = None

        # the build_state as communicated to GitHub:
        # 'failure', 'success', 'pending'
        self.intended_github_status = self.github_status_from_build_state()
        self.last_known_github_status = None

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

    async def authorized(self, dbpool):
        if self.author in {user.gh_username for user in AUTHORIZED_USERS}:
            return True

        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT * from authorized_shas WHERE sha = %s;', self.source_sha)
                row = await cursor.fetchone()
                return row is not None

    def merge_priority(self):
        # passed > unknown > failed
        if self.source_sha_failed is None:
            source_sha_failed_prio = 1
        else:
            source_sha_failed_prio = 0 if self.source_sha_failed else 2

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

    def github_status_from_build_state(self):
        if self.build_state == 'failure' or self.build_state == 'error':
            return 'failure'
        if self.build_state == 'success' and self.batch.attributes['target_sha'] == self.target_branch.sha:
            return 'success'
        return 'pending'

    async def post_github_status(self, gh_client, gh_status):
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
            'state': gh_status,
            'target_url': target_url,
            # FIXME improve
            'description': gh_status,
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
    def _hail_github_status_from_statuses(statuses_json):
        statuses = statuses_json["statuses"]
        hail_status = [s for s in statuses if s["context"] == GITHUB_STATUS_CONTEXT]
        n_hail_status = len(hail_status)
        if n_hail_status == 0:
            return None
        if n_hail_status == 1:
            return hail_status[0]['state']
        raise ValueError(
            f'github sent multiple status summaries for our one '
            f'context {GITHUB_STATUS_CONTEXT}: {hail_status}\n\n{statuses_json}'
        )

    async def _update_last_known_github_status(self, gh):
        if self.source_sha:
            source_sha_json = await gh.getitem(
                f'/repos/{self.target_branch.branch.repo.short_str()}/commits/{self.source_sha}/status'
            )
            last_known_github_status = PR._hail_github_status_from_statuses(source_sha_json)
            if last_known_github_status != self.last_known_github_status:
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

    async def _start_build(self, dbpool, batch_client):
        assert await self.authorized(dbpool)

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

            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read(), scope='test')

            log.info(f'creating test batch for {self.number}')
            batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'test': '1',
                    'source_branch': self.source_branch.short_str(),
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha,
                },
                callback=CALLBACK_URL,
            )
            config.build(batch, self, scope='test')
            batch = await batch.submit()
            self.batch = batch
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            # FIXME save merge failure output for UI
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
    async def is_invalidated_batch(batch, dbpool):
        assert batch is not None
        async with dbpool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('SELECT * from invalidated_batches WHERE batch_id = %s;', batch.id)
                row = await cursor.fetchone()
                return row is not None

    async def _update_batch(self, batch_client, dbpool):
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
            if await self.is_invalidated_batch(b, dbpool):
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

    async def _heal(self, batch_client, dbpool, on_deck, gh):
        # can't merge target if we don't know what it is
        if self.target_branch.sha is None:
            return

        if self.source_sha:
            if self.intended_github_status != self.last_known_github_status:
                log.info(f'Intended github status for {self.short_str()} is: {self.intended_github_status}')
                log.info(f'Last known github status for {self.short_str()} is: {self.last_known_github_status}')
                await self.post_github_status(gh, self.intended_github_status)
                self.last_known_github_status = self.intended_github_status

        if not await self.authorized(dbpool):
            return

        if not self.batch or (on_deck and self.batch.attributes['target_sha'] != self.target_branch.sha):
            if on_deck or self.target_branch.n_running_batches < 8:
                self.target_branch.n_running_batches += 1
                async with repos_lock:
                    await self._start_build(dbpool, batch_client)

    def is_up_to_date(self):
        return self.batch is not None and self.target_branch.sha == self.batch.attributes['target_sha']

    def is_mergeable(self):
        return (
            self.review_state == 'approved'
            and self.build_state == 'success'
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
        return f'''
{clone_or_fetch_script(self.target_branch.branch.repo.url)}

git remote add {shq(self.source_branch.repo.short_str())} {shq(self.source_branch.repo.url)} || true

time retry git fetch -q {shq(self.source_branch.repo.short_str())}
git checkout {shq(self.target_branch.sha)}
git merge {shq(self.source_sha)} -m 'merge PR'
'''


class WatchedBranch(Code):
    def __init__(self, index, branch, deployable, mergeable):
        self.index = index
        self.branch = branch
        self.deployable = deployable
        self.mergeable = mergeable

        self.prs: Optional[Dict[str, PR]] = None
        self.sha = None

        # success, failure, pending
        self.deploy_batch = None
        self._deploy_state = None

        self.updating = False
        self.github_changed = True
        self.batch_changed = True
        self.state_changed = True

        self.n_running_batches = None

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
            dbpool = app['dbpool']

            while self.github_changed or self.batch_changed or self.state_changed:
                if self.github_changed:
                    self.github_changed = False
                    await self._update_github(gh)

                if self.batch_changed:
                    self.batch_changed = False
                    await self._update_batch(batch_client, dbpool)

                if self.state_changed:
                    self.state_changed = False
                    await self._heal(batch_client, dbpool, gh)
                    if self.mergeable:
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

        new_prs: Dict[str, PR] = {}
        async for gh_json_pr in gh.getiter(f'/repos/{repo_ss}/pulls?state=open&base={self.branch.name}'):
            number = gh_json_pr['number']
            if self.prs is not None and number in self.prs:
                pr = self.prs[number]
                pr.update_from_gh_json(gh_json_pr)
            else:
                pr = PR.from_gh_json(gh_json_pr, self)
            new_prs[number] = pr
        if self.prs is not None:
            for number, pr in self.prs.items():
                if number not in new_prs:
                    pr.decrement_pr_metric()
        self.prs = new_prs

        for pr in new_prs.values():
            await pr.assign_gh_reviewer_if_requested(gh)

        for pr in new_prs.values():
            await pr._update_github(gh)

    async def _update_deploy(self, batch_client):
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
@**Daniel Goldstein**
state: {self.deploy_state}
branch: {self.branch.short_str()}
sha: {self.sha}
url: {url}
'''
                    send_zulip_deploy_failure_message(deploy_failure_message)
                self.state_changed = True

    async def _heal_deploy(self, batch_client):
        assert self.deployable

        if not self.sha:
            return

        if self.deploy_batch is None or (self.deploy_state and self.deploy_batch.attributes['sha'] != self.sha):
            async with repos_lock:
                await self._start_deploy(batch_client)

    async def _update_batch(self, batch_client, dbpool):
        log.info(f'update batch {self.short_str()}')

        if self.deployable:
            await self._update_deploy(batch_client)

        for pr in self.prs.values():
            await pr._update_batch(batch_client, dbpool)

    async def _heal(self, batch_client, dbpool, gh):
        log.info(f'heal {self.short_str()}')

        if self.deployable:
            await self._heal_deploy(batch_client)

        merge_candidate = None
        merge_candidate_pri = None
        for pr in self.prs.values():
            # merge candidate if up-to-date build passing, or
            # pending but haven't failed
            if pr.review_state == 'approved' and (pr.build_state == 'success' or not pr.source_sha_failed):
                pri = pr.merge_priority()
                is_authorized = await pr.authorized(dbpool)
                if is_authorized and (not merge_candidate or pri > merge_candidate_pri):
                    merge_candidate = pr
                    merge_candidate_pri = pri
        if merge_candidate:
            log.info(f'merge candidate {merge_candidate.number}')

        self.n_running_batches = sum(1 for pr in self.prs.values() if pr.batch and not pr.build_state)

        for pr in self.prs.values():
            await pr._heal(batch_client, dbpool, pr == merge_candidate, gh)

        # cancel orphan builds
        running_batches = batch_client.list_batches(
            f'!complete ' f'!open ' f'test=1 ' f'target_branch={self.branch.short_str()} ' f'user:ci'
        )
        seen_batch_ids = set(pr.batch.id for pr in self.prs.values() if pr.batch and hasattr(pr.batch, 'id'))
        async for batch in running_batches:
            if batch.id not in seen_batch_ids:
                attrs = batch.attributes
                log.info(f'cancel batch {batch.id} for {attrs["pr"]} {attrs["source_sha"]} => {attrs["target_sha"]}')
                await batch.cancel()

    async def _start_deploy(self, batch_client):
        # not deploying
        assert not self.deploy_batch or self.deploy_state

        self.deploy_batch = None
        self.deploy_state = None

        deploy_batch = None
        try:
            repo_dir = self.repo_dir()
            await check_shell(
                f'''
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
'''
            )
            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read(), requested_step_names=DEPLOY_STEPS, scope='deploy')

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
@**Daniel Goldstein**
branch: {self.branch.short_str()}
sha: {self.sha}
Deploy config failed to build with exception:
```python
{e}
```
'''
                send_zulip_deploy_failure_message(deploy_failure_message)
                raise
            deploy_batch = await deploy_batch.submit()
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

    def checkout_script(self):
        return f'''
{clone_or_fetch_script(self.branch.repo.url)}

git checkout {shq(self.sha)}
'''


class UnwatchedBranch(Code):
    def __init__(self, branch, sha, userdata, extra_config=None):
        self.branch = branch
        self.user = userdata['username']
        self.namespace = userdata['namespace_name']
        self.sha = sha
        self.extra_config = extra_config

        self.deploy_batch = None

    def short_str(self):
        return f'br-{self.branch.repo.owner}-{self.branch.repo.name}-{self.branch.name}'

    def repo_dir(self):
        return f'repos/{self.branch.repo.short_str()}'

    def config(self):
        config = {
            'checkout_script': self.checkout_script(),
            'branch': self.branch.name,
            'repo': self.branch.repo.short_str(),
            'repo_url': self.branch.repo.url,
            'sha': self.sha,
            'user': self.user,
        }
        if self.extra_config is not None:
            config.update(self.extra_config)
        return config

    async def deploy(self, batch_client, steps, excluded_steps=()):
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
            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(
                    self, f.read(), scope='dev', requested_step_names=steps, excluded_step_names=excluded_steps
                )

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
            deploy_batch = await deploy_batch.submit()
            self.deploy_batch = deploy_batch
            return deploy_batch.id
        finally:
            if deploy_batch and not self.deploy_batch and isinstance(deploy_batch, Batch):
                log.info(f'deleting partially created deploy batch {deploy_batch.id}')
                await deploy_batch.delete()

    def checkout_script(self):
        return f'''
{clone_or_fetch_script(self.branch.repo.url)}

git checkout {shq(self.sha)}
'''
