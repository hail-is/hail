import sys
import asyncio
import aiohttp
import timeit
import traceback
import numpy as np


async def run(i, times):
    try:
        async with aiohttp.ClientSession() as session:
            start = timeit.default_timer()
            print(f'{i}: start')
            async with session.get('https://notebook.hail.is/') as resp:
                assert resp.status == 200, await resp.text()
                await resp.text()
            data = aiohttp.FormData()
            data.add_field(name='password', value='hail2019')
            data.add_field(name='image', value='hands-on-with-hail')
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
    assert len(sys.argv) == 2
    n = int(sys.argv[1])
    times = []

    async def f():
        await asyncio.gather(*(run(i, times) for i in range(n)))
    asyncio.run(f())
    print(f'successes: {len(times)}, mean time: {sum(times) / n}')
    print(f'histogram:\n{np.histogram(times, density=True)}')
