from typing import Optional

GITHUB_CLONE_URL = 'https://github.com/'
GITHUB_STATUS_CONTEXT = 'ci-test'


class User:
    def __init__(self, gh_username: str, hail_username: Optional[str]):
        self.gh_username = gh_username
        self.hail_username = hail_username


AUTHORIZED_USERS = [
    User('danking', 'dking'),
    User('cseed', 'cseed'),
    User('konradjk', 'konradk'),
    User('jigold', 'jigold'),
    User('patrick-schultz', 'pschultz'),
    User('lfrancioli', None),
    User('tpoterba', 'tpoterba'),
    User('chrisvittal', 'cvittal'),
    User('catoverdrive', 'wang'),
    User('johnc1231', 'johnc'),
    User('nawatts', None),
    User('mkveerapen', None),
    User('Dania-Abuhijleh', None),
    User('bw2', None),
    User('pwc2', 'pcumming'),
    User('lgruen', None),
    User('CDiaz96', 'carolin'),
    User('daniel-goldstein', 'dgoldste'),
]
