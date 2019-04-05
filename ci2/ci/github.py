from constants import GITHUB_CLONE_URL, SHA_LENGTH
import json


class Repo(object):
    def __init__(self, owner, name):
        assert isinstance(owner, str)
        assert isinstance(name, str)
        self.owner = owner
        self.name = name
        self.url = f'{GITHUB_CLONE_URL}{owner}/{name}.git'
        self.qname = self.short_str()

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


class FQBranch(object):
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
    def from_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'repo' in d, d
        assert 'name' in d, d
        return FQBranch(Repo.from_json(d['repo']), d['name'])

    def to_dict(self):
        return {'repo': self.repo.to_dict(), 'name': self.name}


class FQSHA(object):
    def __init__(self, branch, sha):
        assert isinstance(branch, FQBranch)
        assert isinstance(sha, str)
        self.branch = branch
        self.sha = sha

    def __eq__(self, other):
        return self.branch == other.brach and self.sha == other.sha

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.branch, self.sha))

    @staticmethod
    def from_short_str(s):
        pieces = s.split(":")
        assert len(pieces) == 3, f'{pieces} {s}'
        return FQSHA(FQBranch(Repo.from_short_str(pieces[0]),
                              pieces[1]),
                     pieces[2])

    def short_str(self, sha_length=SHA_LENGTH):
        if sha_length:
            return f'{self.branch.short_str()}:{self.sha[:sha_length]}'
        else:
            return f'{self.brach.short_str()}:{self.sha}'

    @staticmethod
    def from_gh_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'repo' in d, d
        assert 'ref' in d, d
        assert 'sha' in d, d
        return FQSHA(FQBranch(Repo.from_gh_json(d['repo']), d['ref']), d['sha'])

    def __str__(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'brach' in d, d
        assert 'sha' in d, d
        return FQSHA(FQBranch.from_json(d['branch']), d['sha'])

    def to_dict(self):
        return {'brach': self.ref.to_dict(), 'sha': self.sha}


class PR(object):
    def __init__(self, number, title, source, target):
        self.source = source
        self.target = target
        self.number = number
        self.title = title
        self.state = None

    @staticmethod
    def from_gh_json(gh_json):
        return PR(gh_json['number'], gh_json['title'],
                  FQSHA.from_gh_json(gh_json['head']),
                  FQSHA.from_gh_json(gh_json['base']))

    def refresh_from_gh_json(self, gh_json):
        assert gh_json['number'] == self.number
        self.title = gh_json['title']
        self.source = FQSHA.from_gh_json(gh_json['head'])
        self.target = FQSHA.from_gh_json(gh_json['base'])

    async def refresh(self, gh):
        latest_state_by_login = {}
        async for review in gh.getiter(f'/repos/{self.target.branch.repo.short_str()}/pulls/{self.number}/reviews'):
            login = review['user']['login']
            state = review['state']
            # reviews is chronological, so later ones are newer statuses
            if state == 'APPROVED' or state == 'CHANGES_REQUESTED':
                latest_state_by_login[login] = state

        total_state = 'pending'
        for login, state in latest_state_by_login.items():
            if (state == 'CHANGES_REQUESTED'):
                total_state = 'changes_requested'
                break
            elif (state == 'DISMISSED'):
                total_state = 'pending'
                break
            elif (state == 'APPROVED'):
                total_state = 'approved'

        self.state = total_state

    def mark_closed(self):
        pass


class WatchedBranch(object):
    def __init__(self, branch):
        self.branch = branch
        self.prs = {}
        self.sha = None
        self.in_refresh = False

    async def refresh(self, gh):
        if self.in_refresh:
            return
        self.in_refresh = True

        repo_ss = self.branch.repo.short_str()

        branch_gh_json = await gh.getitem(f'/repos/{repo_ss}/git/refs/heads/{self.branch.name}')
        self.sha = branch_gh_json['object']['sha']

        seen = set()
        async for gh_json_pr in gh.getiter(f'/repos/{repo_ss}/pulls?state=open&base={self.branch.name}'):
            number = gh_json_pr['number']
            pr = self.prs.get(number)
            if pr:
                pr.refresh_from_gh_json(gh_json_pr)
            else:
                pr = PR.from_gh_json(gh_json_pr)
                self.prs[number] = pr
            await pr.refresh(gh)
            seen.add(number)

        for pr in self.prs.values():
            if pr.number not in seen:
                del self.prs[pr.number]
                pr.mark_closed()

        self.in_refresh = False
