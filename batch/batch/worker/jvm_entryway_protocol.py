import asyncio
import logging
import struct

log = logging.getLogger('jvm_entryway_protocol')


def write_int(writer: asyncio.StreamWriter, v: int):
    writer.write(struct.pack('>i', v))


def write_long(writer: asyncio.StreamWriter, v: int):
    writer.write(struct.pack('>q', v))


def write_bytes(writer: asyncio.StreamWriter, b: bytes):
    n = len(b)
    write_int(writer, n)
    writer.write(b)


def write_str(writer: asyncio.StreamWriter, s: str):
    write_bytes(writer, s.encode('utf-8'))


class EndOfStream(Exception):
    pass


async def read(reader: asyncio.StreamReader, n: int) -> bytes:
    b = bytearray()
    left = n
    while left > 0:
        t = await reader.read(left)
        if not t:
            log.warning(f'unexpected EOS, Java violated protocol ({b})')
            raise EndOfStream()
        left -= len(t)
        b.extend(t)
    return b


async def read_byte(reader: asyncio.StreamReader) -> int:
    b = await read(reader, 1)
    return b[0]


async def read_bool(reader: asyncio.StreamReader) -> bool:
    return await read_byte(reader) != 0


async def read_int(reader: asyncio.StreamReader) -> int:
    b = await read(reader, 4)
    return struct.unpack('>i', b)[0]


async def read_long(reader: asyncio.StreamReader) -> int:
    b = await read(reader, 8)
    return struct.unpack('>q', b)[0]


async def read_bytes(reader: asyncio.StreamReader) -> bytes:
    n = await read_int(reader)
    return await read(reader, n)


async def read_str(reader: asyncio.StreamReader) -> str:
    b = await read_bytes(reader)
    return b.decode('utf-8')
