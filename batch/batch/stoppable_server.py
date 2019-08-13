import asyncio
import logging
import sys

from aiohttp import web


class StoppableServer:
    def __init__(self, app, host, port):
        self.app = app
        self.host = host
        self.port = port
        self.stop_queue = asyncio.Queue()
        self.log = logging.getLogger('stoppable_server')

    async def stop(self, exit_code):
        await self.stop_queue.put(exit_code)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        self.log.info(f'serving {self.host}:{self.port}')
        await site.start()
        exit_code = await self.stop_queue.get()
        await runner.cleanup()
        return exit_code

    def run(self):
        loop = asyncio.get_event_loop()
        exit_code = 2
        try:
            exit_code = loop.run_until_complete(self.start())
        except:  # pylint: disable=W0702
            self.log.exception(f'caught server exception')
        finally:
            self.log.info(f'shutting down loop')
            loop.close()
            self.log.info(f'exit code {exit_code}')
            sys.exit(exit_code)
