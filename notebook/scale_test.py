import sys
import asyncio
import aiohttp
import timeit
import traceback
import numpy as np
import argparse

async def run(i, times, class_key, image):
    try:
        async with aiohttp.ClientSession() as session:
            start = timeit.default_timer()
            print(f'{i}: start')
            async with session.get('https://notebook.hail.is/') as resp:
                assert resp.status == 200, await resp.text()
                await resp.text()
            data = aiohttp.FormData()
            data.add_field(name='password', value=class_key)
            data.add_field(name='image', value=image)
            print(f'{i}: new')
            async with session.post('https://notebook.hail.is/new', data=data) as resp:
                assert resp.status == 200, await resp.text()
                await resp.text()
            print(f'{i}: wait')
            redirected_url = None
            async with session.ws_connect('wss://notebook.hail.is/waitws') as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        print(f'{i}: received message {msg}')
                        redirected_url = msg.data
                        await ws.close()
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise ValueError(f'{i}: failure! {msg.data}')
                    else:
                        raise ValueError(f'{i}: {msg}')
            print(f'{i}: redirect: {redirected_url}')
            async with session.get(redirected_url, max_redirects=30) as resp:
                assert resp.status == 200, await resp.text()
                await resp.text()
            elapsed = timeit.default_timer() - start
            print(f'{i}: done {elapsed}')
            times.append(elapsed)
    except aiohttp.client_exceptions.TooManyRedirects as e:
        print(f'{i}: failed due to {e} {e.request_info} {repr(traceback.format_exc())} {e.history}')
    except Exception as e:
        print(f'{i}: failed due to {e} {repr(traceback.format_exc())}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scale test notebook.hail.is.')
    parser.add_argument('n', type=int, help='number of notebooks to start')
    parser.add_argument('class_key', type=str, help='class key')
    parser.add_argument('image', type=str, help='image name')
    args = parser.parse_args()
    times = []

    async def f():
        await asyncio.gather(*(run(i, times, args.class_key, args.image)
                               for i in range(args.n)))
    asyncio.run(f())
    print(f'successes: {len(times)}, mean time: {sum(times) / args.n}')
    print(f'histogram:\n{np.histogram(times, density=True)}')
