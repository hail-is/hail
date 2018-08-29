from http_helper import get_repo
import re

clone_url_to_repo = re.compile('https://github.com/([^/]+)/([^/]+).git')


def owner_and_repo_from_url(url):
    m = clone_url_to_repo.match(url)
    assert m and m.lastindex and m.lastindex == 2, f'{m} {url}'
    return (m[1], m[2])


def repo_from_url(url):
    (owner, repo) = owner_and_repo_from_url(url)
    return owner + '/' + repo


def url_from_repo(repo):
    return f'https://github.com/{repo}.git'


def open_pulls(target_repo):
    return get_repo(target_repo.qname, 'pulls?state=open', status_code=200)


def overall_review_state(reviews):
    latest_state_by_login = {}
    for review in reviews:
        login = review['user']['login']
        state = review['state']
        # reviews is chronological, so later ones are newer statuses
        latest_state_by_login[login] = state
    total_state = 'pending'
    for login, state in latest_state_by_login.items():
        if (state == 'CHANGES_REQUESTED'):
            total_state = 'changes_requested'
            break
        elif (state == 'APPROVED'):
            total_state = 'approved'

    return {'state': total_state, 'reviews': latest_state_by_login}
