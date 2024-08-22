import os
from typing import List, Optional

GITHUB_CLONE_URL = 'https://github.com/'
GITHUB_STATUS_CONTEXT = os.environ["HAIL_CI_GITHUB_CONTEXT"]
SERVICES_TEAM = 'Services'
COMPILER_TEAM = 'Compiler'
TEAMS = [SERVICES_TEAM, COMPILER_TEAM]


# Update July 2024:
# Numbers have dwindled lately and the compiler/services split is now untenable.
# For the purposes of the CI reviewer randomiser, anyone on the hail team is
# fair game.
HAIL_TEAM = TEAMS


class User:
    def __init__(self, gh_username: str, hail_username: Optional[str] = None, teams: Optional[List[str]] = None):
        self.gh_username = gh_username
        self.hail_username = hail_username
        self.teams = teams if teams is not None else []


AUTHORIZED_USERS = [
    User('chrisvittal', 'cvittal', HAIL_TEAM),
    User('cjllanwarne', 'chrisl', HAIL_TEAM),
    User('ehigham', 'ehigham', HAIL_TEAM),
    User('illusional', 'mfrankli'),
    User('iris-garden', 'irademac', HAIL_TEAM),
    User('jkgoodrich', 'jgoodric'),
    User('konradjk', 'konradk'),
    User('patrick-schultz', 'pschultz', HAIL_TEAM),
]
