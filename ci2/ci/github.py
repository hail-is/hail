import secrets
from shlex import quote as shq
import json
import asyncio
from .log import log
from .constants import GITHUB_CLONE_URL
from .utils import CalledProcessError, check_shell, check_shell_output
from .build import BuildConfiguration

repos_lock = asyncio.Lock()


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


class PR:
    def __init__(self, number, title, source_repo, source_sha, target_branch):
        self.number = number
        self.title = title
        self.source_repo = source_repo
        self.source_sha = source_sha
        self.target_branch = target_branch

        # one of pending, changes_requested, approve
        self.review_state = None

        self.sha = None
        self.batch = None
        self.build_state = None
        self.target_branch.batch_changed = True

    def update_from_gh_json(self, gh_json):
        assert self.number == gh_json['number']
        self.title = gh_json['title']

        head = gh_json['head']
        new_source_sha = head['sha']
        if self.source_sha != new_source_sha:
            self.source_sha = new_source_sha
            self.sha = None
            self.batch = None
            self.build_state = None
            self.target_branch.batch_changed = True

        self.source_repo = Repo.from_gh_json(head['repo'])

    @staticmethod
    def from_gh_json(gh_json, target_branch):
        head = gh_json['head']
        return PR(gh_json['number'], gh_json['title'], Repo.from_gh_json(head['repo']), head['sha'], target_branch)

    def config(self):
        assert self.sha is not None
        target_repo = self.target_branch.branch.repo
        return {
            'number': self.number,
            'source_repo': self.source_repo.short_str(),
            'source_repo_url': self.source_repo.url,
            'source_sha': self.source_sha,
            'target_repo': target_repo.short_str(),
            'target_repo_url': target_repo.url,
            'target_sha': self.target_branch.sha,
            'sha': self.sha
        }

    async def refresh_review_state(self, gh):
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

        self.review_state = review_state

    async def start_build(self, batch_client):
        assert not self.batch

        try:
            log.info(f'merging for {self.number}')
            repo_dir = f'repos/{self.target_branch.branch.repo.short_str()}'

            merge_script = f'''
set -ex
if [ ! -d "{shq(repo_dir)}" ]; then
  mkdir -p {shq(repo_dir)}
  git clone {shq(self.target_branch.branch.repo.url)} {shq(repo_dir)}

  git -C {shq(repo_dir)} config user.email hail-ci@example.com
  git -C {shq(repo_dir)} config user.name hail-ci
else
  git -C {shq(repo_dir)} reset --merge
  git -C {shq(repo_dir)} fetch origin
fi

git -C {shq(repo_dir)} remote add {shq(self.source_repo.short_str())} {shq(self.source_repo.url)} || true

git -C {shq(repo_dir)} fetch {shq(self.source_repo.short_str())}
git -C {shq(repo_dir)} checkout {shq(self.target_branch.sha)}
git -C {shq(repo_dir)} merge {shq(self.source_sha)} -m 'merge PR'
'''
            await check_shell(merge_script)

            sha_out, _ = await check_shell_output(
                f'git -C {shq(repo_dir)} rev-parse HEAD')
            self.sha = sha_out.decode('utf-8').strip()

            with open(f'{repo_dir}/build.yaml', 'r') as f:
                config = BuildConfiguration(self, f.read())
        except (CalledProcessError, FileNotFoundError) as e:
            log.exception(f'could not open build.yaml due to {e}')
            self.build_state = 'merge_failure'
            self.target_branch.batch_changed = True
            return

        batch = None
        try:
            log.info(f'creating batch for {self.number}')
            batch = await batch_client.create_batch(
                attributes={
                    'token': secrets.token_hex(16),
                    'target_branch': self.target_branch.branch.short_str(),
                    'pr': str(self.number),
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha
                })
            await config.build(batch, self)
            await batch.close()
            self.batch = batch
        finally:
            if batch and not self.batch:
                await batch.cancel()

    async def heal(self, batch_client, run, seen_batch_ids):
        if self.build_state is not None:
            if self.batch:
                seen_batch_ids.add(self.batch.id)
            return

        if self.batch is None:
            batches = await batch_client.list_batches(
                complete=False,
                attributes={
                    'source_sha': self.source_sha,
                    'target_sha': self.target_branch.sha
                })

            # FIXME
            def batch_was_cancelled(batch):
                status = batch.status()
                return any(j['state'] == 'Cancelled' for j in status['jobs'])

            # we might be returning to a commit that was only partially tested
            batches = [b for b in batches if not batch_was_cancelled(b)]

            # should be at most one batch
            if len(batches) > 0:
                self.batch = batches[0]
            elif run:
                async with repos_lock:
                    await self.start_build(batch_client)

        if self.batch:
            seen_batch_ids.add(self.batch.id)
            status = await self.batch.status()
            if all(j['state'] == 'Complete' for j in status['jobs']):
                self.build_state = 'success' if all(j['exit_code'] == 0 for j in status['jobs']) else 'failure'
                self.batch_changed = True


class WatchedBranch:
    def __init__(self, branch):
        self.branch = branch
        self.prs = None
        self.sha = None
        self.updating = False
        self.github_changed = True
        self.batch_changed = True

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
            return
        self.updating = True

        gh = app['github_client']
        batch_client = app['batch_client']

        while self.github_changed or self.batch_changed:
            if self.github_changed:
                self.github_changed = False
                await self.refresh(gh)

            if self.batch_changed:
                self.batch_changed = False
                await self.heal(batch_client)

        self.updating = False

    async def refresh(self, gh):
        repo_ss = self.branch.repo.short_str()

        branch_gh_json = await gh.getitem(f'/repos/{repo_ss}/git/refs/heads/{self.branch.name}')
        new_sha = branch_gh_json['object']['sha']
        if new_sha != self.sha:
            self.sha = new_sha
            if self.prs:
                for pr in self.prs.values():
                    pr.sha = None
                    pr.batch = None
                    pr.build_state = None
            self.batch_changed = True

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
            await pr.refresh_review_state(gh)

    async def heal(self, batch_client):
        # FIXME merge
        merge_candidate = None
        for pr in self.prs.values():
            if pr.review_state == 'approved' and pr.build_state is None:
                merge_candidate = pr
                break
        if merge_candidate:
            log.info(f'merge candidate {merge_candidate.number}')

        running_batches = await batch_client.list_batches(
            complete=False,
            attributes={
                'target_branch': self.branch.short_str()
            })

        seen_batch_ids = set()
        for pr in self.prs.values():
            await pr.heal(batch_client, (merge_candidate is None or pr == merge_candidate), seen_batch_ids)

        for batch in running_batches:
            if batch.id not in seen_batch_ids:
                await batch.cancel()
