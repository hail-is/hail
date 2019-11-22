import math
import time
import argparse
import asyncio
import logging
import numpy as np
import struct

import dbuf.client

import hailtop.utils as utils

log = logging.getLogger('dbuf_scale_test')


async def write(server, id, data, args):
    start = time.time()
    async with dbuf.client.DBufClient(server, id, max_bufsize=args.bufsize*1024*1024-1) as client:
        writer = client.start_write()
        for i in range(args.reqs):
            await writer.write(data[i])
        keys = await writer.keys()
        return keys, time.time() - start


async def read(server, id, args, keys):
    start = time.time()
    async with dbuf.client.DBufClient(server, id, max_bufsize=args.bufsize*1024*1024-1) as client:
        data = await client.getmany(keys)
    return data, time.time() - start


async def main():
    parser = argparse.ArgumentParser(description='distributed buffer scale test')
    parser.add_argument('cluster_leader', type=str, help='cluster leader name, e.g. dbuf-0.dbuf')
    parser.add_argument('n', type=int, help='number of clients')
    parser.add_argument('bufsize', type=int, help='bufsize in MB')
    parser.add_argument('size', type=int, help='number of bytes to send per request')
    parser.add_argument('reqs', type=int, help='number of requests to send')
    args = parser.parse_args()

    n = args.n

    start = time.time()
    async with dbuf.client.DBufClient(args.cluster_leader) as client:
        id = await client.create()

        workers = await client.get_workers()
        server = [workers[i % len(workers)] for i in range(n)]

        def bytearray_with_index(i):
            b = bytearray(args.size)
            struct.pack_into('l', b, 0, i)
            return b
        data = [bytearray_with_index(i) for i in range(n * args.reqs)]
        grouped_data = list(utils.grouped(data, args.reqs))

        keys, times = utils.unzip(await asyncio.gather(
            *[write(server[i], id, grouped_data[i], args) for i in range(n)]))

        end = time.time()
        duration = end - start
        print(f'write aggregate-throughput: {n * args.size * args.reqs / duration / 1024 / 1024 / 1024 : 0.3f} GiB/s')
        keys = [x for xs in keys for x in xs]
        indices = list(range(len(keys)))
        np.random.shuffle(indices)
        keys = [keys[i] for i in indices]
        data = [data[i] for i in indices]
        keys = list(utils.grouped(keys, args.reqs))
        data = list(utils.grouped(data, args.reqs))

        start = time.time()

        data2, times = utils.unzip(await asyncio.gather(
            *[read(server[i], id, args, keys[i]) for i in range(n)]))

        end = time.time()
        duration = end - start

        print(f'read aggregate-throughput: {n * args.size * args.reqs / duration / 1024 / 1024 / 1024 : 0.3f} GiB/s')

        await client.delete()

        assert len(data) == len(data2), f'{len(data)} {len(data2)}'
        assert data == data2, [(i, j,
                                struct.unpack_from("l", x) if len(x) > 8 else x,
                                struct.unpack_from("l", y) if len(y) > 8 else y)
                               for i, (xs, ys) in enumerate(zip(data, data2))
                               for j, (x, y) in enumerate(zip(xs, ys))
                               if x != y]


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
