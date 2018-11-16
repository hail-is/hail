import requests


# https://github.com/requests/requests/pull/4234
def raise_on_failure(response, max_body_text=500):
    if 400 <= response.status_code < 600:
        blame = 'Client' if response.status_code < 500 else 'Server'
        raise requests.HTTPError(
            f'{response.status_code} {blame} Error for '
            f'url {response.url}. {short_body(response, max_body_text)}',
            response=response
        )


def short_body(response, max_body_text):
    if isinstance(response.text, str):
        if len(response.text) < max_body_text:
            return response.text
        return response.text[:max_body_text-3] + '...'
    return ''
