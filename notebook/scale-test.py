import argparse
import asyncio
import logging
import math
import time

import aiohttp
import numpy as np

from hailtop.auth import hail_credentials
from hailtop.config import get_deploy_config
from hailtop.hail_logging import configure_logging
from hailtop.httpx import client_session

configure_logging()
log = logging.getLogger('nb-scale-test')

deploy_config = get_deploy_config()


def get_cookie(session, name):
    for cookie in session.cookie_jar:
        if cookie.key == name:
            return cookie.value
    return None


async def run(args, i):
    headers = await hail_credentials(authorize_target=False).auth_headers()

    async with client_session() as session:
        # make sure notebook is up
        async with session.get(deploy_config.url('workshop', ''), headers=headers) as resp:
            await resp.text()

        log.info(f'{i} loaded notebook home page')

        # log in as workshop guest
        # get csrf token
        async with session.get(deploy_config.url('workshop', '/login'), headers=headers) as resp:
            pass

        data = aiohttp.FormData()
        data.add_field(name='name', value=args.workshop)
        data.add_field(name='password', value=args.password)
        data.add_field(name='_csrf', value=get_cookie(session, '_csrf'))
        async with session.post(deploy_config.url('workshop', '/login'), data=data, headers=headers) as resp:
            pass

        log.info(f'{i} logged in')

        # create notebook
        # get csrf token
        async with session.get(deploy_config.url('workshop', '/notebook'), headers=headers) as resp:
            pass

        data = aiohttp.FormData()
        data.add_field(name='_csrf', value=get_cookie(session, '_csrf'))
        async with session.post(deploy_config.url('workshop', '/notebook'), data=data, headers=headers) as resp:
            pass

        log.info(f'{i} created notebook')

        start = time.time()

        # wait for notebook ready
        ready = False
        attempt = 0
        # 5 attempts overkill, should only take 2: Scheduling => Running => Ready
        while not ready and attempt < 5:
            async with session.ws_connect(
                deploy_config.url('workshop', '/notebook/wait', base_scheme='ws'), headers=headers
            ) as ws:
                async for msg in ws:
                    if msg.data == '1':
                        ready = True
            attempt += 1

        end = time.time()
        duration = end - start

        log.info(f'{i} notebook state {ready} duration {duration}')

        # delete notebook
        # get csrf token
        async with session.get(deploy_config.url('workshop', '/notebook'), headers=headers) as resp:
            pass

        data = aiohttp.FormData()
        data.add_field(name='_csrf', value=get_cookie(session, '_csrf'))
        async with session.post(deploy_config.url('workshop', '/notebook/delete'), data=data, headers=headers) as resp:
            pass

        log.info(f'{i} notebook delete, done.')

    return duration, ready


async def main():
    parser = argparse.ArgumentParser(description='Notebook scale test.')
    parser.add_argument('n', type=int, help='number of notebooks to start')
    parser.add_argument('workshop', type=str, help='workshop name')
    parser.add_argument('password', type=str, help='workshop password')
    args = parser.parse_args()

    n = args.n
    d = int(math.log10(n)) + 1
    outcomes = await asyncio.gather(*[run(args, str(i).zfill(d)) for i in range(n)])

    times = []
    for duration, ready in outcomes:
        if ready:
            times.append(duration)

    print(f'successes: {len(times)} / {n} = {len(times) / n}')
    print(f'mean time: {sum(times) / n}')
    print(f'quantiles min/50%/95%/99%/max: {np.quantile(times, [0.0, .5, .95, .99, 1.0])}')
    print(f'histogram:\n{np.histogram(times, density=True)}')


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
