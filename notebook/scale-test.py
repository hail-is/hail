import asyncio
import logging
import aiohttp
from gear import configure_logging
from hailtop.auth import service_auth_headers
from hailtop.config import get_deploy_config

configure_logging()
log = logging.getLogger('nb-scale-test')

deploy_config = get_deploy_config()


def get_cookie(session, name):
    for cookie in session.cookie_jar:
        if cookie.key == name:
            return cookie.value
    return None


async def main():
    headers = service_auth_headers(deploy_config, 'notebook', authorize_target=False)

    async with aiohttp.ClientSession(raise_for_status=True) as session:
        # make sure notebook is up
        async with session.get(
                deploy_config.url('notebook', ''),
                headers=headers) as resp:
            await resp.text()

        log.info('loaded notebook home page')

        # log in as workshop guest
        # get csrf token
        async with session.get(
                deploy_config.url('notebook', '/workshop/login'),
                headers=headers) as resp:
            pass

        data = aiohttp.FormData()
        data.add_field(name='name', value='p')
        data.add_field(name='password', value='q')
        data.add_field(name='_csrf', value=get_cookie(session, '_csrf'))
        async with session.post(
                deploy_config.url('notebook', '/workshop/login'),
                data=data,
                headers=headers) as resp:
            pass

        log.info('logged in')

        # create notebook
        # get csrf token
        async with session.get(
                deploy_config.url('notebook', '/workshop/notebook'),
                headers=headers) as resp:
            pass

        data = aiohttp.FormData()
        data.add_field(name='_csrf', value=get_cookie(session, '_csrf'))
        async with session.post(
                deploy_config.url('notebook', '/workshop/notebook'),
                data=data,
                headers=headers) as resp:
            pass

        log.info('created notebook')

        # wait for notebook ready
        ready = False
        attempt = 0
        while attempt < 5:
            async with session.ws_connect(
                    deploy_config.url('notebook', '/workshop/notebook/wait', base_scheme='ws'),
                    headers=headers) as ws:
                async for msg in ws:
                    print(msg.data)
                    if msg.data == '1':
                        ready = True
                        break
            attempt += 1

        log.info(f'notebook state: {ready}')

        # delete notebook
        # get csrf token
        async with session.get(
                deploy_config.url('notebook', '/workshop/notebook'),
                headers=headers) as resp:
            pass

        data = aiohttp.FormData()
        data.add_field(name='_csrf', value=get_cookie(session, '_csrf'))
        async with session.post(
                deploy_config.url('notebook', '/workshop/notebook/delete'),
                data=data,
                headers=headers) as resp:
            pass

        log.info('notebook delete, done.')


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
