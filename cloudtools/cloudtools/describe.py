import json
from zlib import decompress, MAX_WBITS
from subprocess import check_output
from statistics import median, mean, stdev
from collections import OrderedDict

SECTION_SEPARATOR = '-'*40
IDENT = ' '*4

def parse_schema(s):
    def parse_type(s, end_delimiter, element_type):
        keys = []
        values = []
        i = 0
        while i < len(s):
            if s[i] == end_delimiter:
                if s[:i]:
                    values.append(s[:i])
                if element_type in ['Array', 'Set', 'Dict']:
                    return {'type': element_type, 'value': values}, s[i + 1:]
                else:
                    return {'type': element_type, 'value': OrderedDict(zip(keys, values))}, s[i + 1:]
            elif s[i] == ':':
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
    return parse_type(s[start_schema_index+1:], "}", s[:start_schema_index])[0]


def type_str(t, depth=1):
    NAME_MAP = {
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
            else:
                return "{}<{}>".format(
                    e['type'].lower(),
                    ", ".join([element_str(x) for x in e['value']])
                )
        else:
            return NAME_MAP.get(e, e).lower().replace('(', '<').replace(')', '>')

    return "\n".join(
        "{}'{}': {}".format(IDENT * depth, k, element_str(v))
        for k, v in t.items()
    )

def key_str(k):
    if isinstance(k, dict):
        return '[{}]'.format(', '.join([key_str(x) for x in k['value']]))
    else:
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

def main(args):

    command = ['gsutil'] if args.file.startswith('gs://') else []

    j = json.loads(
        decompress(
            check_output(command + ['cat', args.file + '/metadata.json.gz']),
            16+MAX_WBITS
        )
    )

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
    try:
        check_output(command + ['ls', args.file + '/_SUCCESS'])
    except:
        print("\033[;1m\033[1;31mCould not find _SUCCESS for file: {}\nThis file will not work.\033[0m".format(args.file))
