import json
import yaml
import aiohttp
import csv
from typing import List, Dict, Callable
import tabulate
import io

TableData = List[Dict[str, str]]
TABLE_FORMAT_OPTIONS = ['json', 'yaml', 'csv', *tabulate.tabulate_formats]


def get_batch_if_exists(client, id):
    try:
        return client.get_batch(id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            return None
        raise cle


def get_job_if_exists(client, batch_id, job_id):
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
