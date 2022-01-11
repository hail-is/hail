import json
import yaml
import aiohttp
import csv
import sys
from typing import List, Dict, Callable
import tabulate

LoD = List[Dict[str, str]]


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


def separated_format(lod: LoD, delim: str) -> None:
    writer = csv.writer(sys.stdout, delimiter=delim)
    # construct header dictionary from keys of lod
    header = {k: k for (k, v) in lod[0].items()}
    writer.writerow(header)
    # for each of the elements in the list of dicts, write its values as a list to the writer
    for d in lod:
        writer.writerow([v for (k, v) in d.items()])


def table_format(lod: LoD, tablefmt: str) -> None:
    print(tabulate.tabulate(lod, headers='keys', tablefmt=tablefmt))


# POST-CONDITION: the returned formatters are only guaranteed to work on non-empty lists of dictionaries
def make_formatter(name: str) -> Callable[[List[Dict[str, str]]], None]:
    if name == "json":
        return lambda lod: print(json.dumps(lod, indent=2))
    if name == "yaml":
        return lambda lod: print(yaml.dump(lod))
    if name == "csv":
        return lambda lod: separated_format(lod, ',')
    if name == "tsv":
        return lambda lod: separated_format(lod, '\t')
    if name == "orgtbl":
        return lambda lod: table_format(lod, 'orgtbl')
    if name == "grid":
        return lambda lod: table_format(lod, 'grid')
    raise ValueError(f'unknown format {name}')
