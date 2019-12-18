import random
import time
import argparse
import asyncio
import logging
import numpy as np
import struct

import dbuf.client

import hailtop.utils as utils

log = logging.getLogger('dbuf_scale_test')


async def write(data, args, client):
    start = time.time()
    writer = await client.start_write()
    for i in range(args.reqs):
        await writer.write(data[i])
    keys = await writer.finish()
    return keys, time.time() - start


async def read(args, keys, client):
    start = time.time()
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

    max_bufsize = args.bufsize * 1024 * 1024

    print('dbuf scale test')
    print(args)
    async with dbuf.client.DBufClient(args.cluster_leader, max_bufsize=max_bufsize, rng=random.Random(0)) as client:
        print('creating session')
        await client.create()

        def bytearray_with_index(i):
            b = bytearray(args.size)
            struct.pack_into('l', b, 0, i)
            return b
        print('creating data')
        data = [bytearray_with_index(i) for i in range(n * args.reqs)]
        data_for_worker = list(utils.grouped(args.reqs, data))

        print(f'starting test')
        start = time.time()
        keys, times = utils.unzip(await asyncio.gather(
            *[write(data_for_worker[i], args, client) for i in range(n)]))
        end = time.time()
        duration = end - start
        print(f'write aggregate-throughput: {n * args.size * args.reqs / duration / 1024 / 1024 / 1024 : 0.3f} GiB/s')

        keys = [x for xs in keys for x in xs]
        indices = list(range(len(keys)))
        np.random.shuffle(indices)
        keys = [keys[i] for i in indices]
        data = [data[i] for i in indices]
        keys = list(utils.grouped(args.reqs, keys))
        data = list(utils.grouped(args.reqs, data))

        start = time.time()
        data2, times = utils.unzip(await asyncio.gather(
            *[read(args, keys[i], client) for i in range(n)]))
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
