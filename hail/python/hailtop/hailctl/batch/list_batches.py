import aiohttp
from hailtop.batch_client.client import BatchClient


def main(args, passthrough_args):
    session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
    bc = BatchClient(session, url="http://batch.hail.is")

    print(bc.list_batches())