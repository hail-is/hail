from constants import GITHUB_API_URL
from environment import oauth_token
import re
import requests


class BadStatus(Exception):
    def __init__(self, data, status_code):
        Exception.__init__(self, str(data))
        self.data = data
        self.status_code = status_code


def patch_repo(repo,
               url,
               headers=None,
               json=None,
               data=None,
               status_code=None,
               json_response=True,
               token=oauth_token):
    return verb_repo(
        'patch',
        repo,
        url,
        headers=headers,
        json=json,
        data=data,
        status_code=status_code,
        json_response=json_response,
        token=token)


def post_repo(repo,
              url,
              headers=None,
              json=None,
              data=None,
              status_code=None,
              json_response=True,
              token=oauth_token):
    return verb_repo(
        'post',
        repo,
        url,
        headers=headers,
        json=json,
        data=data,
        status_code=status_code,
        json_response=json_response,
        token=token)


def get_repo(repo,
             url,
             headers=None,
             status_code=None,
             json_response=True,
             token=oauth_token):
    return verb_repo(
        'get',
        repo,
        url,
        headers=headers,
        status_code=status_code,
        json_response=json_response,
        token=token)


def put_repo(repo,
             url,
             headers=None,
             json=None,
             data=None,
             status_code=None,
             json_response=True,
             token=oauth_token):
    return verb_repo(
        'put',
        repo,
        url,
        headers=headers,
        json=json,
        data=data,
        status_code=status_code,
        json_response=json_response,
        token=token)


def get_github(url, headers=None, status_code=None):
    return verb_github('get', url, headers=headers, status_code=status_code)


def verb_repo(verb,
              repo,
              url,
              headers=None,
              json=None,
              data=None,
              status_code=None,
              json_response=True,
              token=oauth_token):
    return verb_github(
        verb,
        f'repos/{repo}/{url}',
        headers=headers,
        json=json,
        data=data,
        status_code=status_code,
        json_response=json_response,
        token=token)


def implies(antecedent, consequent):
    return not antecedent or consequent


verbs = set(['post', 'put', 'get', 'patch'])


def verb_github(verb,
                url,
                headers=None,
                json=None,
                data=None,
                status_code=None,
                json_response=True,
                token=oauth_token):
    if isinstance(status_code, int):
        status_codes = [status_code]
    else:
        status_codes = status_code
    assert verb in verbs, f'{verb} {verbs}'
    assert implies(verb == 'post' or verb == 'put',
                   json is not None or data is not None)
    assert implies(verb == 'get', json is None and data is None)
    if headers is None:
        headers = {}
    if 'Authorization' in headers:
        raise ValueError('Header already has Authorization? ' + str(headers))
    headers['Authorization'] = 'token ' + token
    full_url = f'{GITHUB_API_URL}{url}'
    if verb == 'get':
        r = requests.get(full_url, headers=headers, timeout=5)
        if json_response:
            output = r.json()
            if 'Link' in r.headers:
                assert isinstance(output, list), output
                link = r.headers['Link']
                url = github_link_header_to_maybe_next(link)
                while url is not None:
                    r = requests.get(url, headers=headers, timeout=5)
                    link = r.headers['Link']
                    output.extend(r.json())
                    url = github_link_header_to_maybe_next(link)
        else:
            output = r.text
    else:
        if verb == 'post':
            r = requests.post(
                full_url,
                headers=headers,
                data=data,
                json=json,
                timeout=5)
        elif verb == 'put':
            r = requests.put(
                full_url,
                headers=headers,
                data=data,
                json=json,
                timeout=5)
        elif verb == 'patch':
            r = requests.patch(
                full_url,
                headers=headers,
                data=data,
                json=json,
                timeout=5)
        if json_response:
            output = r.json()
        else:
            output = r.text
    if status_codes and r.status_code not in status_codes:
        raise BadStatus({
            'method': verb,
            'endpoint': full_url,
            'status_code': {
                'actual': r.status_code,
                'expected': status_codes
            },
            'message': 'github error',
            'data': data,
            'json': json,
            'github_json': output
        },
                        r.status_code)
    else:
        if isinstance(status_code, list):
            return (output, r.status_code)
        else:
            return output


github_link = re.compile(r'\s*<(http.+page=[0-9]+)>; rel="([A-z]+)"\s*')


def github_link_header_to_maybe_next(link):
    # I cannot find rigorous documentation on the format, but this seems to
    # work?
    link_texts = link.split(',')
    links = {}
    for t in link_texts:
        m = github_link.match(t)
        assert m is not None, f'{m} {t}'
        links[m[2]] = m[1]
    return links.get('next', None)
