from typing import Optional, List

GITHUB_CLONE_URL = 'https://github.com/'
GITHUB_STATUS_CONTEXT = 'ci-test'
TEAMS = ['Services', 'Compiler']


class User:
    # pylint: disable=dangerous-default-value
    def __init__(self, gh_username: str, hail_username: Optional[str] = None, teams: List[str] = []):
        self.gh_username = gh_username
        self.hail_username = hail_username
        self.teams = teams


AUTHORIZED_USERS = [
    User('danking', 'dking', ['Services']),
    User('cseed', 'cseed'),
    User('konradjk', 'konradk'),
    User('jigold', 'jigold', ['Services']),
    User('patrick-schultz', 'pschultz', ['Compiler']),
    User('lfrancioli'),
    User('tpoterba', 'tpoterba', ['Compiler']),
    User('chrisvittal', 'cvittal', ['Compiler']),
    User('catoverdrive', 'wang', ['Services', 'Compiler']),
    User('johnc1231', 'johnc', ['Compiler']),
    User('nawatts'),
    User('mkveerapen'),
    User('Dania-Abuhijleh'),
    User('bw2'),
    User('pwc2', 'pcumming'),
    User('lgruen'),
    User('CDiaz96', 'carolin', ['Services']),
    User('daniel-goldstein', 'dgoldste', ['Services']),
]
