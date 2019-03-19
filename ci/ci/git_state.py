from .constants import GITHUB_CLONE_URL, SHA_LENGTH
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


class FQRef(object):
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
        return FQRef(Repo.from_short_str(pieces[0]), pieces[1])

    def short_str(self):
        return f'{self.repo.short_str()}:{self.name}'

    @staticmethod
    def from_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'repo' in d, d
        assert 'name' in d, d
        return FQRef(Repo.from_json(d['repo']), d['name'])

    def to_dict(self):
        return {'repo': self.repo.to_dict(), 'name': self.name}


class FQSHA(object):
    def __init__(self, ref, sha):
        assert isinstance(ref, FQRef)
        assert isinstance(sha, str)
        self.ref = ref
        self.sha = sha

    def __eq__(self, other):
        return self.ref == other.ref and self.sha == other.sha

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.ref, self.sha))

    @staticmethod
    def from_short_str(s):
        pieces = s.split(":")
        assert len(pieces) == 3, f'{pieces} {s}'
        return FQSHA(FQRef(Repo.from_short_str(pieces[0]),
                           pieces[1]),
                     pieces[2])

    def short_str(self, sha_length=SHA_LENGTH):
        if sha_length:
            return f'{self.ref.short_str()}:{self.sha[:sha_length]}'
        else:
            return f'{self.ref.short_str()}:{self.sha}'

    @staticmethod
    def from_gh_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'repo' in d, d
        assert 'ref' in d, d
        assert 'sha' in d, d
        return FQSHA(FQRef(Repo.from_gh_json(d['repo']), d['ref']), d['sha'])

    def __str__(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(d):
        assert isinstance(d, dict), f'{type(d)} {d}'
        assert 'ref' in d, d
        assert 'sha' in d, d
        return FQSHA(FQRef.from_json(d['ref']), d['sha'])

    def to_dict(self):
        return {'ref': self.ref.to_dict(), 'sha': self.sha}
