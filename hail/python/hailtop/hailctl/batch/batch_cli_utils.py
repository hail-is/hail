import json
from enum import Enum
import yaml
import csv
from typing import List, Dict, Callable
import tabulate
import io

from typer import Option as Opt

from typing_extensions import Annotated as Ann

TableData = List[Dict[str, str]]
TABLE_FORMAT_OPTIONS = ['json', 'yaml', 'csv', *tabulate.tabulate_formats]


class StructuredFormat(str, Enum):
    YAML = 'yaml'
    JSON = 'json'

    def __str__(self):
        return self.value


StructuredFormatOption = Ann[StructuredFormat, Opt('--output', '-o')]


class StructuredFormatPlusText(str, Enum):
    TEXT = 'text'
    YAML = 'yaml'
    JSON = 'json'

    def __str__(self):
        return self.value


StructuredFormatPlusTextOption = Ann[StructuredFormatPlusText, Opt('--output', '-o')]


class ExtendedOutputFormat(str, Enum):
    YAML = 'yaml'
    JSON = 'json'
    GRID = 'grid'

    def __str__(self):
        return self.value


ExtendedOutputFormatOption = Ann[ExtendedOutputFormat, Opt('--output', '-o')]


def get_batch_if_exists(client, id):
    import aiohttp.client_exceptions  # pylint: disable=import-outside-toplevel

    try:
        return client.get_batch(id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            return None
        raise cle


def get_job_if_exists(client, batch_id, job_id):
    import aiohttp.client_exceptions  # pylint: disable=import-outside-toplevel

    try:
        return client.get_job(batch_id, job_id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            return None
        raise cle


def bool_string_to_bool(bool_string):
    if bool_string in ["True", "true", "t"]:
        return True
    if bool_string in ['False', 'false', 'f']:
        return False
    raise ValueError("Input could not be resolved to a bool")


def separated_format(table_data: TableData, delim: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delim)
    assert len(table_data) > 0
    header = table_data[0].keys()
    writer.writerow(header)
    for d in table_data:
        writer.writerow(d.values())

    contents = output.getvalue()
    output.close()
    return contents


def table_format(table_data: TableData, tablefmt: str) -> str:
    return tabulate.tabulate(table_data, headers='keys', tablefmt=tablefmt)


# POST-CONDITION: the returned formatters are only guaranteed to work on non-empty lists of dictionaries
def make_formatter(name: str) -> Callable[[TableData], str]:
    assert name in TABLE_FORMAT_OPTIONS, f'unknown format: {name}'

    if name == "json":
        return lambda table_data: json.dumps(table_data, indent=2)
    if name == "yaml":
        return yaml.dump
    if name == "csv":
        return lambda table_data: separated_format(table_data, ',')

    assert name in tabulate.tabulate_formats, f'unknown tabulate format {name}'
    return lambda table_data: table_format(table_data, name)
