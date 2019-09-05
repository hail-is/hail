import secrets
from shlex import quote as shq
import json
import logging
import asyncio
import concurrent.futures
import aiohttp
import gidgethub
from .constants import GITHUB_CLONE_URL, AUTHORIZED_USERS
from .environment import SELF_HOSTNAME
from .utils import check_shell, check_shell_output
from .build import BuildConfiguration, Code

repos_lock = asyncio.Lock()

log = logging.getLogger('ci')


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


HIGH_PRIORITY = 'prio:high'
STACKED_PR = 'stacked PR'
WIP = 'WIP'

DO_NOT_MERGE = {STACKED_PR, WIP}


class PR(Code):
    def __init__(self, number, title, source_repo, source_sha, target_branch, author, labels):
        self.number = number
        self.title = title
        self.source_repo = source_repo
        self.source_sha = source_sha
        self.target_branch = target_branch
        self.author = author
        self.labels = labels

        # pending, changes_requested, approve
        self.review_state = None

        self.sha = None
        self.batch = None
        self.source_sha_failed = None

        # error, success, failure
        self._build_state = None

        # don't need to set github_changed because we are refreshing github
        self.target_branch.batch_changed = True
        self.target_branch.state_changed = True

    @property
    def build_state(self):
        return self._build_state

    @build_state.setter
    def build_state(self, new_state):
        if new_state != self._build_state:
            self._build_state = new_state

    async def authorized(self, dbpool):
        if self.author in AUTHORIZED_USERS:
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

        return (HIGH_PRIORITY in self.labels,
                all(label not in DO_NOT_MERGE for label in self.labels),
                source_sha_failed_prio,
                # oldest first
                - self.number)

    def short_str(self):
        return f'pr-{self.number}'

    def update_from_gh_json(self, gh_json):
        assert self.number == gh_json['number']
        self.title = gh_json['title']
        self.author = gh_json['user']['login']

        new_labels = {l['name'] for l in gh_json['labels']}
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
            self.build_state = None
            self.target_branch.batch_changed = True
            self.target_branch.state_changed = True

        self.source_repo = Repo.from_gh_json(head['repo'])

    @staticmethod
    def from_gh_json(gh_json, target_branch):
        head = gh_json['head']
        return PR(gh_json['number'],
                  gh_json['title'],
                  Repo.from_gh_json(head['repo']),
                  head['sha'],
                  target_branch,
                  gh_json['user']['login'],
                  {l['name'] for l in gh_json['labels']})

    def repo_dir(self):
        return self.target_branch.repo_dir()

    def config(self):
        assert self.sha is not None
        target_repo = self.target_branch.branch.repo
        return {
            'checkout_script': self.checkout_script(),
            'number': self.number,
            'source_repo': self.source_repo.short_str(),
            'source_repo_url': self.source_repo.url,
            'source_sha': self.source_sha,
            'target_repo': target_repo.short_str(),
            'target_repo_url': target_repo.url,
            'target_sha': self.target_branch.sha,
            'sha': self.sha
        }

    def github_status(self):
        if self.build_state == 'failure' or self.build_state == 'error':
            return 'failure'
        if (self.build_state == 'success' and
                self.batch.attributes['target_sha'] == self.target_branch.sha):
            return 'success'
        return 'pending'

    async def post_github_status(self, gh_client, gh_status):
        assert self.source_sha is not None

        log.info(f'{self.short_str()}: notify github state: {gh_status}')
        data = {
            'state': gh_status,
            # FIXME should be this build, not the pr
            'target_url': f'https://ci.hail.is/watched_branches/{self.target_branch.index}/pr/{self.number}',
            # FIXME improve
            'description': gh_status,
            'context': 'ci-test'
        }
        try:
            await gh_client.post(
                f'/repos/{self.target_branch.branch.repo.short_str()}/statuses/{self.source_sha}',
                data=data)
        except gidgethub.HTTPException as e:
            log.info(f'{self.short_str()}: notify github of build state failed due to exception: {e}')
        except aiohttp.client_exceptions.ClientResponseError as e:
            log.error(f'{self.short_str()}: Unexpected exception in post to github: {e}')

    async def _update_github_review_state(self, gh):
        latest_state_by_login = {}
        async for review in gh.getiter(f'/repos/{self.target_branch.branch.repo.short_str()}/pulls/{self.number}/reviews'):
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
            self.review_state = review_state
            self.target_branch.state_changed = True

    async def _start_build(self, dbpool, batch_client):
        assert await self.authorized(dbpool)

        # clear current batch
        self.batch = None
        self.build_state = None

        batch = None
        try:
            log.info(f'merging for {self.number}')
            repo_dir = self.repo_dir()
            await check_shell(f'''
set -ex
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
''')

            sha_out, _ = await check_shell_output(
                f'git -C {shq(repo_dir)} rev-parse HEAD')
            self.sha = sha_out.decode('utf-8').strip()

            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read(), scope='test')

            log.info(f'creating test batch for {self.number}')
            batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'test': '1',
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha
                },
                callback=f'http://{SELF_HOSTNAME}/api/v1alpha/batch_callback')
            config.build(batch, self, scope='test')
            batch = await batch.submit()
            self.batch = batch
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            log.exception(f'could not start build due to {e}')

            # FIXME save merge failure output for UI
            self.batch = MergeFailureBatch(
                e,
                attributes={
                    'test': '1',
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha,
                })
            self.build_state = 'error'
            self.source_sha_failed = True
            self.target_branch.state_changed = True
        finally:
            if batch and not self.batch:
                log.info(f'cancelling partial test batch {batch.id}')
                await batch.cancel()

    async def _update_batch(self, batch_client):
        if self.build_state:
            assert self.batch
            return

        if self.batch is None:
            # find the latest non-cancelled batch for source
            attrs = {
                'test': '1',
                'target_branch': self.target_branch.branch.short_str(),
                'source_sha': self.source_sha
            }
            batches = await batch_client.list_batches(attributes=attrs)

            min_batch = None
            failed = None
            for b in batches:
                try:
                    s = await b.status()
                except Exception as err:
                    log.info(f'failed to get the status for batch {b.id} due to error: {err}')
                    raise
                if s['state'] != 'cancelled':
                    if min_batch is None or b.id > min_batch.id:
                        min_batch = b

                    if s['state'] == 'failure':
                        failed = True
                    elif failed is None:
                        failed = False
            self.batch = min_batch
            self.source_sha_failed = failed

        if self.batch:
            try:
                status = await self.batch.status()
            except aiohttp.client_exceptions.ClientResponseError as exc:
                if exc.status == 404:
                    log.info(f'batch {self.batch.id} was deleted by someone')
                    self.batch = None
                    self.build_state = None
                    return
                raise exc
            if status['complete']:
                if status['state'] == 'success':
                    self.build_state = 'success'
                else:
                    self.build_state = 'failure'
                    self.source_sha_failed = True
                self.target_branch.state_changed = True

    async def _heal(self, batch_client, dbpool, on_deck):
        # can't merge target if we don't know what it is
        if self.target_branch.sha is None:
            return

        if not await self.authorized(dbpool):
            return

        if (not self.batch or
                (on_deck and self.batch.attributes['target_sha'] != self.target_branch.sha)):

            if on_deck or self.target_branch.n_running_batches < 4:
                self.target_branch.n_running_batches += 1
                async with repos_lock:
                    await self._start_build(dbpool, batch_client)

    def is_up_to_date(self):
        return self.batch is not None and self.target_branch.sha == self.batch.attributes['target_sha']

    def is_mergeable(self):
        return (self.review_state == 'approved' and
                self.build_state == 'success' and
                self.is_up_to_date() and
                all(label not in DO_NOT_MERGE for label in self.labels))

    async def merge(self, gh):
        try:
            await gh.put(f'/repos/{self.target_branch.branch.repo.short_str()}/pulls/{self.number}/merge',
                         data={
                             'merge_method': 'squash',
                             'sha': self.source_sha
                         })
            return True
        except (gidgethub.HTTPException, aiohttp.client_exceptions.ClientResponseError) as e:
            log.info(f'merge {self.target_branch.branch.short_str()} {self.number} failed due to exception: {e}')
        return False

    def checkout_script(self):
        return f'''
if [ ! -d .git ]; then
  time git clone {shq(self.target_branch.branch.repo.url)} .

  git config user.email ci@hail.is
  git config user.name ci
else
  git reset --merge
  time git fetch -q origin
fi

git remote add {shq(self.source_repo.short_str())} {shq(self.source_repo.url)} || true

time git fetch -q {shq(self.source_repo.short_str())}
git checkout {shq(self.target_branch.sha)}
git merge {shq(self.source_sha)} -m 'merge PR'
'''


class WatchedBranch(Code):
    def __init__(self, index, branch, deployable):
        self.index = index
        self.branch = branch
        self.deployable = deployable

        self.prs = None
        self.sha = None

        # success, failure, pending
        self.deploy_batch = None
        self._deploy_state = None

        self.updating = False
        self.github_changed = True
        self.batch_changed = True
        self.state_changed = True

        self.statuses = {}
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
            'sha': self.sha
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
                    await self.try_to_merge(gh)

                if self.batch_changed:
                    self.batch_changed = False
                    await self._update_batch(batch_client)
                    await self.try_to_merge(gh)

                if self.state_changed:
                    self.state_changed = False
                    await self._heal(batch_client, dbpool)
            await self.update_statuses(gh)
        finally:
            log.info(f'update done {self.short_str()}')
            self.updating = False

    async def try_to_merge(self, gh):
        await self.update_statuses(gh)
        for pr in self.prs.values():
            if pr.is_mergeable():
                if await pr.merge(gh):
                    self.github_changed = True
                    self.sha = None
                    self.state_changed = True
                    return

    async def update_statuses(self, gh):
        new_statuses = {}
        for pr in self.prs.values():
            if pr.source_sha:
                gh_status = pr.github_status()
                if pr.source_sha not in self.statuses or self.statuses[pr.source_sha] != gh_status:
                    await pr.post_github_status(gh, gh_status)
                new_statuses[pr.source_sha] = gh_status
        self.statuses = new_statuses

    async def _update_github(self, gh):
        log.info(f'update github {self.short_str()}')

        repo_ss = self.branch.repo.short_str()

        branch_gh_json = await gh.getitem(f'/repos/{repo_ss}/git/refs/heads/{self.branch.name}')
        new_sha = branch_gh_json['object']['sha']
        if new_sha != self.sha:
            log.info(f'{self.branch.short_str()} sha changed: {self.sha} => {new_sha}')
            self.sha = new_sha
            self.state_changed = True

        new_prs = {}
        async for gh_json_pr in gh.getiter(f'/repos/{repo_ss}/pulls?state=open&base={self.branch.name}'):
            number = gh_json_pr['number']
            if self.prs is not None and number in self.prs:
                pr = self.prs[number]
                pr.update_from_gh_json(gh_json_pr)
            else:
                pr = PR.from_gh_json(gh_json_pr, self)
            new_prs[number] = pr
        self.prs = new_prs

        for pr in new_prs.values():
            await pr._update_github_review_state(gh)

    async def _update_deploy(self, batch_client):
        assert self.deployable

        if self.deploy_state:
            assert self.deploy_batch
            return

        if self.deploy_batch is None:
            running_deploy_batches = await batch_client.list_batches(
                complete=False,
                attributes={
                    'deploy': '1',
                    'target_branch': self.branch.short_str()
                })
            if running_deploy_batches:
                self.deploy_batch = max(running_deploy_batches, key=lambda b: b.id)
            else:
                deploy_batches = await batch_client.list_batches(
                    attributes={
                        'deploy': '1',
                        'target_branch': self.branch.short_str(),
                        'sha': self.sha
                    })
                if deploy_batches:
                    self.deploy_batch = max(deploy_batches, key=lambda b: b.id)

        if self.deploy_batch:
            try:
                status = await self.deploy_batch.status()
            except aiohttp.client_exceptions.ClientResponseError as exc:
                log.info(f'Could not update deploy_batch status due to exception {exc}, setting deploy_batch to None')
                self.deploy_batch = None
                return
            if status['complete']:
                if status['state'] == 'success':
                    self.deploy_state = 'success'
                else:
                    self.deploy_state = 'failure'
                self.state_changed = True

    async def _heal_deploy(self, batch_client):
        assert self.deployable

        if not self.sha:
            return

        if (self.deploy_batch is None or
                (self.deploy_state and self.deploy_batch.attributes['sha'] != self.sha)):
            async with repos_lock:
                await self._start_deploy(batch_client)

    async def _update_batch(self, batch_client):
        log.info(f'update batch {self.short_str()}')

        if self.deployable:
            await self._update_deploy(batch_client)

        for pr in self.prs.values():
            await pr._update_batch(batch_client)

    async def _heal(self, batch_client, dbpool):
        log.info(f'heal {self.short_str()}')

        if self.deployable:
            await self._heal_deploy(batch_client)

        merge_candidate = None
        merge_candidate_pri = None
        for pr in self.prs.values():
            # merge candidate if up-to-date build passing, or
            # pending but haven't failed
            if (pr.review_state == 'approved' and
                    (pr.build_state == 'success' or not pr.source_sha_failed)):
                pri = pr.merge_priority()
                is_authorized = await pr.authorized(dbpool)
                if is_authorized and (not merge_candidate or pri > merge_candidate_pri):
                    merge_candidate = pr
                    merge_candidate_pri = pri
        if merge_candidate:
            log.info(f'merge candidate {merge_candidate.number}')

        self.n_running_batches = sum(1 for pr in self.prs.values() if pr.batch and not pr.build_state)

        for pr in self.prs.values():
            await pr._heal(batch_client, dbpool, pr == merge_candidate)

        # cancel orphan builds
        running_batches = await batch_client.list_batches(
            complete=False,
            attributes={
                'test': '1',
                'target_branch': self.branch.short_str()
            })
        seen_batch_ids = set(pr.batch.id for pr in self.prs.values() if pr.batch and hasattr(pr.batch, 'id'))
        for batch in running_batches:
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
            await check_shell(f'''
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
''')
            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read(), scope='deploy')

            log.info(f'creating deploy batch for {self.branch.short_str()}')
            deploy_batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'deploy': '1',
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha
                },
                callback=f'http://{SELF_HOSTNAME}/api/v1alpha/batch_callback')
            config.build(deploy_batch, self, scope='deploy')
            deploy_batch = await deploy_batch.submit()
            self.deploy_batch = deploy_batch
        except concurrent.futures.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            log.exception(f'could not start deploy due to {e}')
            self.deploy_batch = MergeFailureBatch(
                e,
                attributes={
                    'deploy': '1',
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha
                })
            self.deploy_state = 'checkout_failure'
        finally:
            if deploy_batch and not self.deploy_batch:
                log.info(f'cancelling partial deploy batch {deploy_batch.id}')
                await deploy_batch.cancel()

    def checkout_script(self):
        return f'''
if [ ! -d .git ]; then
  time git clone {shq(self.branch.repo.url)} .

  git config user.email ci@hail.is
  git config user.name ci
else
  git reset --merge
  time git fetch -q origin
fi

git checkout {shq(self.sha)}
'''


class UnwatchedBranch(Code):
    def __init__(self, branch, sha, userdata):
        self.branch = branch
        self.user = userdata['username']
        self.namespace = userdata['namespace_name']
        self.sha = sha

        self.deploy_batch = None

    def short_str(self):
        return f'br-{self.branch.repo.owner}-{self.branch.repo.name}-{self.branch.name}'

    def repo_dir(self):
        return f'repos/{self.branch.repo.short_str()}'

    def config(self):
        return {
            'checkout_script': self.checkout_script(),
            'branch': self.branch.name,
            'repo': self.branch.repo.short_str(),
            'repo_url': self.branch.repo.url,
            'sha': self.sha,
            'user': self.user
        }

    async def deploy(self, batch_client, steps):
        assert not self.deploy_batch

        deploy_batch = None
        try:
            repo_dir = self.repo_dir()
            await check_shell(f'''
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
''')
            log.info(f'User {self.user} requested these steps for dev deploy: {steps}')
            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read(), scope='dev', requested_step_names=steps)

            log.info(f'creating dev deploy batch for {self.branch.short_str()} and user {self.user}')

            deploy_batch = batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha,
                    'user': self.user
                })
            config.build(deploy_batch, self, scope='dev')
            deploy_batch = await deploy_batch.submit()
            self.deploy_batch = deploy_batch
            return deploy_batch.id
        finally:
            if deploy_batch and not self.deploy_batch:
                log.info(f'cancelling partial deploy batch {deploy_batch.id}')
                await deploy_batch.cancel()

    def checkout_script(self):
        return f'''
if [ ! -d .git ]; then
  time git clone {shq(self.branch.repo.url)} .

  git config user.email ci@hail.is
  git config user.name ci
else
  git reset --merge
  time git fetch -q origin
fi

git checkout {shq(self.sha)}
'''
