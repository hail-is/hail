import aiohttp

def get_batch_if_exists(client, id):
    try:
        return client.get_batch(id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            return None
        raise cle
