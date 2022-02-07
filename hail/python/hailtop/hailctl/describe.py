import argparse
import asyncio
import orjson
import sys
from os import path
from zlib import decompress, MAX_WBITS
from statistics import median, mean, stdev
from collections import OrderedDict

from ..aiotools import aio_contextlib
from ..aiotools.router_fs import RouterAsyncFS


SECTION_SEPARATOR = '-' * 40
IDENT = ' ' * 4


def parse_schema(s):
    def parse_type(s, end_delimiter, element_type):
        keys = []
        values = []
        i = 0
        while i < len(s):
            if s[i] == end_delimiter:
                if s[:i]:
                    values.append(s[:i])
                if element_type in ['Array', 'Set', 'Dict', 'Tuple', 'Interval']:
                    return {'type': element_type, 'value': values}, s[i + 1:]
                return {'type': element_type, 'value': OrderedDict(zip(keys, values))}, s[i + 1:]

            if s[i] == ':':
                keys.append(s[:i])
                s = s[i + 1:]
                i = 0
            elif s[i] == '{':
                struct, s = parse_type(s[i + 1:], '}', s[:i])
                values.append(struct)
                i = 0
            elif s[i] == '[':
                arr, s = parse_type(s[i + 1:], ']', s[:i] if s[:i] else 'Array')
                values.append(arr)
                i = 0
            elif s[i] == ',':
                if s[:i]:
                    values.append(s[:i])
                s = s[i + 1:]
                i = 0
            else:
                i += 1

        raise Exception('End of {} not found'.format(element_type))

    start_schema_index = s.index('{')
    return parse_type(s[start_schema_index + 1:], "}", s[:start_schema_index])[0]


def type_str(t, depth=1):
    name_map = {
        'Boolean': 'bool',
        'String': 'str'
    }

    def element_str(e):
        if isinstance(e, dict):
            if e['type'] == 'Struct':
                return "struct {{\n{}\n{}}}".format(
                    type_str(e['value'], depth + 1),
                    (IDENT * depth)
                )
            return "{}<{}>".format(
                e['type'].lower(),
                ", ".join([element_str(x) for x in e['value']])
            )
        return name_map.get(e, e).lower().replace('(', '<').replace(')', '>')

    return "\n".join(
        "{}'{}': {}".format(IDENT * depth, k, element_str(v))
        for k, v in t.items()
    )


def key_str(k):
    if isinstance(k, dict):
        return '[{}]'.format(', '.join([key_str(x) for x in k['value']]))
    return "'{}'".format(k)


def get_partitions_info_str(j):
    partitions = j['components']['partition_counts']['counts']
    partitions_info = {
        'Partitions': len(partitions),
        'Rows': sum(partitions),
        'Empty partitions': len([p for p in partitions if p == 0])
    }
    if partitions_info['Partitions'] > 1:
        partitions_info.update({
            'Min(rows/partition)': min(partitions),
            'Max(rows/partition)': max(partitions),
            'Median(rows/partition)': median(partitions),
            'Mean(rows/partition)': int(mean(partitions)),
            'StdDev(rows/partition)': int(stdev(partitions))
        })

    return "\n{}".format(IDENT).join(['{}: {}'.format(k, v) for k, v in partitions_info.items()])


def init_parser(parser):
    # arguments with default parameters
    parser.add_argument('file', type=str, help='Path to hail file (either MatrixTable or Table).')
    parser.add_argument('--requester-pays-project-id', '-u', help='Project to be billed for GCS requests.')


def main(args):
    describe_parser = argparse.ArgumentParser(
        prog='hailctl describe',
        description='Describe Hail Matrix Table and Table files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    init_parser(describe_parser)
    if not args:
        describe_parser.print_help()
        sys.exit(0)
    asyncio.get_event_loop().run_until_complete(
        main_after_parsing(
            *describe_parser.parse_known_args(args=args)))


async def main_after_parsing(args, pass_through_args):  # pylint: disable=unused-argument
    gcs_kwargs = {}
    if args.requester_pays_project_id:
        gcs_kwargs['project'] = args.requester_pays_project_id

    async with aio_contextlib.closing(
            RouterAsyncFS(default_scheme='file', gcs_kwargs=gcs_kwargs)) as fs:
        j = orjson.loads(decompress(await fs.read(path.join(args.file, 'metadata.json.gz')),
                                    16 + MAX_WBITS))

        # Get the file schema
        file_schema = parse_schema(j[[k for k in j.keys() if k.endswith('type')][0]])

        # Print file information
        print(SECTION_SEPARATOR)
        print('File Type: {}'.format(file_schema['type']))
        print(IDENT + get_partitions_info_str(j))

        # Print global fields
        print(SECTION_SEPARATOR)
        print('Global fields:')
        print(type_str(file_schema['value']['global']['value']))

        # Print column fields if present
        if 'col' in file_schema['value']:
            print(SECTION_SEPARATOR)
            print('Column fields:')
            print(type_str(file_schema['value']['col']['value']))

        # Print row fields
        print(SECTION_SEPARATOR)
        print('Row fields:')
        print(type_str(file_schema['value']['row']['value']))

        # Print entry fields if present
        if 'entry' in file_schema['value']:
            print(SECTION_SEPARATOR)
            print('Entry fields:')
            print(type_str(file_schema['value']['entry']['value']))

        # Print keys
        print(SECTION_SEPARATOR)
        if 'col_key' in file_schema['value']:
            print("Column key: {}".format(key_str(file_schema['value']['col_key'])))
            print("Row key: {}".format(key_str(file_schema['value']['row_key'])))
        else:
            print("Key: {}".format(key_str(file_schema['value']['key'])))
        print(SECTION_SEPARATOR)

        # Check for _SUCCESS
        if not await fs.exists(path.join(args.file, '_SUCCESS')):
            print(
                "\033[;1m\033[1;31mCould not find _SUCCESS for file: {}\nThis file will not work.\033[0m".format(args.file))
