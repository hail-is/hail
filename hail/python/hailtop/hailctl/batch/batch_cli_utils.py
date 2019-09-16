import aiohttp


def get_batch_if_exists(client, id):
    try:
        return client.get_batch(id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            return None
        raise cle


def get_job_if_exists(client, batch_id, job_id):
    try:
        return client.get_job(batch_id, job_id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            return None
        raise cle


def bool_string_to_bool(bool_string):
    if bool_string in ["True", "true", "t"]:
        return True
    if bool_string in ['False', 'false', 'f']:
        return False
    raise ValueError("Input could not be resolved to a bool")
