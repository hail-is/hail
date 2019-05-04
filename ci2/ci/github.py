import secrets
from shlex import quote as shq
import json
import asyncio
import aiohttp
import gidgethub
from .log import log
from .constants import GITHUB_CLONE_URL
from .environment import SELF_HOSTNAME
from .utils import CalledProcessError, check_shell, check_shell_output, update_batch_status
from .build import BuildConfiguration, Code

repos_lock = asyncio.Lock()


def build_state_is_complete(build_state):
    return build_state in ('success', 'failure')


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
    def __init__(self, attributes=None):
        self.attributes = attributes


class PR(Code):
    def __init__(self, number, title, source_repo, source_sha, target_branch, author, high_prio):
        self.number = number
        self.title = title
        self.source_repo = source_repo
        self.source_sha = source_sha
        self.target_branch = target_branch
        self.author = author
        self.high_prio = high_prio

        # pending, changes_requested, approve
        self.review_state = None

        self.sha = None
        self.batch = None
        self.sha_failed = None

        # merge_failure, success, failure
        self.build_state = None

        # don't need to set github_changed because we are refreshing github
        self.target_branch.batch_changed = True
        self.target_branch.state_changed = True

    def merge_priority(self):
        # passed > unknown > failed
        if self.sha_failed is None:
            sha_failed_prio = 1
        else:
            sha_failed_prio = 0 if self.sha_failed else 2

        return (self.high_prio,
                sha_failed_prio,
                # oldest first
                - self.number)

    def short_str(self):
        return f'pr-{self.number}'

    def update_from_gh_json(self, gh_json):
        assert self.number == gh_json['number']
        self.title = gh_json['title']
        self.author = gh_json['user']['login']

        new_high_prio = any(l['name'] == 'prio:high' for l in gh_json['labels'])
        if new_high_prio != self.high_prio:
            self.high_prio = new_high_prio
            self.target_branch.state_changed = True

        head = gh_json['head']
        new_source_sha = head['sha']
        if self.source_sha != new_source_sha:
            log.info(f'{self.short_str()} source sha changed: {self.source_sha} => {new_source_sha}')
            self.source_sha = new_source_sha
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
                  any(l['name'] == 'prio:high' for l in gh_json['labels']))

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

    def build_is_complete(self):
        return build_state_is_complete(self.build_state)

    async def post_github_status(self, gh):
        assert self.source_sha is not None

        log.info(f'{self.short_str()}: notify github of build state: {self.build_state}')
        if self.build_state is None:
            gh_state = 'pending'
        elif self.build_state == 'success':
            gh_state = 'success'
        else:
            assert self.build_state == 'failure' or self.build_state == 'merge_failure'
            gh_state = 'failure'
        data = {
            'state': gh_state,
            # FIXME should be this build, not the pr
            'target_url': f'https://ci2.hail.is/watched_branches/{self.target_branch.index}/pr/{self.number}',
            'description': self.build_state,
            'context': 'ci-test'
        }
        try:
            url = f'/repos/{self.target_branch.branch.repo.short_str()}/statuses/{self.source_sha}'
            log.info(f'notify github url: {url} {data}')
            await gh.post(f'/repos/{self.target_branch.branch.repo.short_str()}/statuses/{self.source_sha}', data=data)
        except gidgethub.HTTPException as e:
            log.info(f'{self.short_str()}: notify github of build state failed due to exception: {e}')

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
                assert state in ('DISMISSED', 'COMMENTED'), state

        if review_state != self.review_state:
            self.review_state = review_state
            self.target_branch.state_changed = True

    async def _start_build(self, batch_client):
        # clear current batch
        source_sha_changed = self.batch is None or self.batch.attributes['source_sha'] != self.source_sha
        self.batch = None
        if source_sha_changed:
            self.sha_failed = None
        self.build_state = None

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
                config = BuildConfiguration(self, f.read(), deploy=False)
        except (CalledProcessError, FileNotFoundError) as e:
            log.exception(f'could not open build.yaml due to {e}')
            # FIXME save merge failure output for UI
            self.batch = MergeFailureBatch(
                attributes={
                    'test': '1',
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha
                })
            self.build_state = 'merge_failure'
            self.sha_failed = True
            self.target_branch.state_changed = True
            return

        batch = None
        try:
            log.info(f'creating test batch for {self.number}')
            batch = await batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'test': '1',
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha
                },
                callback=f'http://{SELF_HOSTNAME}/batch_callback')
            await config.build(batch, self, deploy=False)
            await batch.close()
            self.batch = batch
        except Exception as e:  # pylint: disable=broad-except
            log.exception(f'could not build {self.number} due to exception {e}')
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
                s = await b.status()
                update_batch_status(s)
                if s['state'] != 'cancelled':
                    if min_batch is None or b.id > min_batch.id:
                        min_batch = b

                    if s['state'] == 'failure':
                        failed = True
                    elif failed is None:
                        failed = False
            self.batch = min_batch
            self.sha_failed = failed

        if self.batch:
            status = await self.batch.status()
            update_batch_status(status)
            if status['complete']:
                if status['state'] == 'success':
                    self.build_state = 'success'
                else:
                    self.build_state = 'failure'
                    self.sha_failed = True
                self.target_branch.state_changed = True

    async def _heal(self, batch_client, on_deck):
        if (not self.batch or
                self.batch.attributes['source_sha'] != self.source_sha or
                (on_deck and self.batch.attributes['target_sha'] != self.target_branch.sha)):

            if on_deck or self.target_branch.n_running_batches < 4:
                self.target_branch.n_running_batches += 1
                async with repos_lock:
                    await self._start_build(batch_client)

    def is_mergeable(self):
        return (self.review_state == 'approved' and
                self.build_state == 'success' and
                self.source_sha == self.batch.attributes['source_sha'] and
                self.target_branch.sha == self.batch.attributes['target_sha'])

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
  time git fetch origin
fi

git remote add {shq(self.source_repo.short_str())} {shq(self.source_repo.url)} || true

time git fetch {shq(self.source_repo.short_str())}
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
        self.deploy_state = None

        self.updating = False
        self.github_changed = True
        self.batch_changed = True
        self.state_changed = True

        self.statuses = {}
        self.n_running_batches = None

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
                    await self._heal(batch_client)

            # update statuses
            new_statuses = {}
            for pr in self.prs.values():
                if pr.source_sha:
                    if pr.source_sha not in self.statuses or self.statuses[pr.source_sha] != pr.build_state:
                        await pr.post_github_status(gh)
                    new_statuses[pr.source_sha] = pr.build_state
            self.statuses = new_statuses
        finally:
            log.info(f'update done {self.short_str()}')
            self.updating = False

    async def try_to_merge(self, gh):
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
                    self.deploy_batch = max(running_deploy_batches, key=lambda b: b.id)

        if self.deploy_batch:
            status = await self.deploy_batch.status()
            update_batch_status(status)
            if status['complete']:
                if status['state'] == 'success':
                    self.deploy_state = 'success'
                else:
                    self.deploy_state = 'failure'
                self.state_changed = True

    async def _heal_deploy(self, batch_client):
        assert self.deployable

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

    async def _heal(self, batch_client):
        log.info(f'heal {self.short_str()}')

        if self.deployable:
            await self._heal_deploy(batch_client)

        merge_candidate = None
        merge_candidate_pri = None
        for pr in self.prs.values():
            # merge candidate if up-to-date build passing, or
            # pending but haven't failed
            if (pr.review_state == 'approved' and
                    ((pr.build_state == 'success' and
                      pr.source_sha == pr.batch.attributes['source_sha']) or
                     pr.sha_failed != True)):
                pri = pr.merge_priority()
                if not merge_candidate or pri > merge_candidate_pri:
                    merge_candidate = pr
                    merge_candidate_pri = pri
        if merge_candidate:
            log.info(f'merge candidate {merge_candidate.number}')

        self.n_running_batches = sum(1 for pr in self.prs.values() if pr.batch and not pr.build_state)

        for pr in self.prs.values():
            await pr._heal(batch_client, pr == merge_candidate)

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

        try:
            repo_dir = self.repo_dir()
            await check_shell(f'''
mkdir -p {shq(repo_dir)}
(cd {shq(repo_dir)}; {self.checkout_script()})
''')
            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read(), deploy=True)
        except (CalledProcessError, FileNotFoundError) as e:
            log.exception(f'could not open build.yaml due to {e}')
            self.deploy_batch = MergeFailureBatch(
                attributes={
                    'deploy': '1',
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha
                })
            self.deploy_state = 'checkout_failure'
            return

        deploy_batch = None
        try:
            log.info(f'creating deploy batch for {self.branch.short_str()}')
            deploy_batch = await batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'deploy': '1',
                    'target_branch': self.branch.short_str(),
                    'sha': self.sha
                },
                callback=f'http://{SELF_HOSTNAME}/batch_callback')
            # FIXME make build atomic
            await config.build(deploy_batch, self, deploy=True)
            await deploy_batch.close()
            self.deploy_batch = deploy_batch
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
  time git fetch origin
fi

git checkout {shq(self.sha)}
'''
