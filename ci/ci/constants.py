import os
from typing import List, Optional

GITHUB_CLONE_URL = 'https://github.com/'
GITHUB_STATUS_CONTEXT = os.environ["HAIL_CI_GITHUB_CONTEXT"]
SERVICES_TEAM = 'Services'
COMPILER_TEAM = 'Compiler'
TEAMS = [SERVICES_TEAM, COMPILER_TEAM]


class User:
    def __init__(self, gh_username: str, hail_username: Optional[str] = None, teams: Optional[List[str]] = None):
        self.gh_username = gh_username
        self.hail_username = hail_username
        self.teams = teams if teams is not None else []


AUTHORIZED_USERS = [
    User('bw2'),
    User('chrisvittal', 'cvittal', [COMPILER_TEAM]),
    User('cseed', 'cseed'),
    User('daniel-goldstein', 'dgoldste', [SERVICES_TEAM]),
    User('danking', 'dking'),
    User('dependabot[bot]'),
    User('jigold', 'jigold', [SERVICES_TEAM]),
    User('jkgoodrich', 'jgoodric'),
    User('konradjk', 'konradk'),
    User('nawatts'),
    User('patrick-schultz', 'pschultz', [COMPILER_TEAM]),
    User('pwc2', 'pcumming'),
    User('tpoterba', 'tpoterba', []),
    User('vladsaveliev', 'vsavelye', []),
    User('illusional', 'mfrankli', []),
    User('iris-garden', 'irademac'),
    User('ehigham', 'ehigham', [COMPILER_TEAM]),
]
