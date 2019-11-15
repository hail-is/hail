import math
import time
import argparse
import asyncio
import logging
import numpy as np
import struct

import dbuf.client

log = logging.getLogger('dbuf_scale_test')


async def write(server, id, data, args, i):
    start = time.time()
    async with dbuf.client.DBufClient(server, id, max_bufsize=args.bufsize*1024*1024-1) as client:
        keys = []
        for i in range(args.reqs):
            keys += await client.append(data[i])
        keys += await client.flush()
        return keys, time.time() - start


async def read(server, id, args, i, keys):
    start = time.time()
    async with dbuf.client.DBufClient(server, id, max_bufsize=args.bufsize*1024*1024-1) as client:
        data = await client.getmany(keys)
    return data, time.time() - start


def grouped(xs, size):
    n = len(xs)
    i = 0
    while i < n:
        yield xs[i:(i+size)]
        i += size


def unzip(xys):
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return xs, ys


async def main():
    parser = argparse.ArgumentParser(description='distributed buffer scale test')
    parser.add_argument('n', type=int, help='number of clients')
    parser.add_argument('bufsize', type=int, help='bufsize in MB')
    parser.add_argument('size', type=int, help='number of bytes to send per request')
    parser.add_argument('reqs', type=int, help='number of requests to send')
    args = parser.parse_args()

    n = args.n
    d = int(math.log10(n)) + 1

    start = time.time()
    async with dbuf.client.DBufClient('http://localhost:5000') as client:
        print(f'create')
        id = await client.create()

        workers = await client.get_workers()
        server = [workers[i % len(workers)] for i in range(n)]

        def bytes(i):
            b = bytearray(args.size)
            struct.pack_into('l', b, 0, i)
            return b
        data = [bytes(i) for i in range(n * args.reqs)]
        grouped_data = list(grouped(data, args.reqs))

        keys, times = unzip(await asyncio.gather(
            *[write(server[i], id, grouped_data[i], args, str(i).zfill(d)) for i in range(n)]))

        end = time.time()
        duration = end - start
        print(f'write aggregate-throughput: {n * args.size * args.reqs / duration / 1024 / 1024 / 1024 : 0.3f} GiB/s')
        keys = [x for xs in keys for x in xs]
        indices = list(range(len(keys)))
        np.random.shuffle(indices)
        keys = [keys[i] for i in indices]
        data = [data[i] for i in indices]
        keys = list(grouped(keys, args.reqs))
        data = list(grouped(data, args.reqs))

        start = time.time()

        data2, times = unzip(await asyncio.gather(
            *[read(server[i], id, args, str(i).zfill(d), keys[i]) for i in range(n)]))

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
