from typing import Optional, List
import os

GITHUB_CLONE_URL = 'https://github.com/'
GITHUB_STATUS_CONTEXT = os.environ["HAIL_CI_GITHUB_CONTEXT"]
SERVICES_TEAM = 'Services'
COMPILER_TEAM = 'Compiler'
TEAMS = [SERVICES_TEAM, COMPILER_TEAM]


class User:
    # pylint: disable=dangerous-default-value
    def __init__(self, gh_username: str, hail_username: Optional[str] = None, teams: List[str] = []):
        self.gh_username = gh_username
        self.hail_username = hail_username
        self.teams = teams


AUTHORIZED_USERS = [
    User('danking', 'dking', [SERVICES_TEAM]),
    User('cseed', 'cseed'),
    User('konradjk', 'konradk'),
    User('jigold', 'jigold', [SERVICES_TEAM]),
    User('patrick-schultz', 'pschultz', [COMPILER_TEAM]),
    User('lfrancioli'),
    User('tpoterba', 'tpoterba', [COMPILER_TEAM]),
    User('chrisvittal', 'cvittal', [COMPILER_TEAM]),
    User('johnc1231', 'johnc', [COMPILER_TEAM]),
    User('nawatts'),
    User('mkveerapen'),
    User('bw2'),
    User('pwc2', 'pcumming'),
    User('lgruen'),
    User('CDiaz96', 'carolin', [SERVICES_TEAM]),
    User('daniel-goldstein', 'dgoldste', [SERVICES_TEAM]),
    User('ammekk', 'ammekk'),
]
