import json
import os
from typing import Optional, Union

import pkg_resources

import hail as hl


def _read_dataset(path: str) -> Union[hl.Table, hl.MatrixTable, hl.linalg.BlockMatrix]:
    if path.endswith('.ht'):
        return hl.read_table(path)
    elif path.endswith('.mt'):
        return hl.read_matrix_table(path)
    elif path.endswith('.bm'):
        return hl.linalg.BlockMatrix.read(path)
    raise ValueError(f'Invalid path: {path}. Can only load datasets with .ht, .mt, or .bm extensions.')


def load_dataset(name: str,
                 version: Optional[str],
                 reference_genome: Optional[str],
                 region: str = 'us',
                 cloud: str = 'gcp') -> Union[hl.Table, hl.MatrixTable, hl.linalg.BlockMatrix]:
    """Load a genetic dataset from Hail's repository.

    Example
    -------
    >>> # Load the gnomAD "HGDP + 1000 Genomes" dense MatrixTable with GRCh38 coordinates.
    >>> mt = hl.experimental.load_dataset(name='gnomad_hgdp_1kg_subset_dense',
    ...                                   version='3.1.2',
    ...                                   reference_genome='GRCh38',
    ...                                   region='us',
    ...                                   cloud='gcp')

    Parameters
    ----------
    name : :class:`str`
        Name of the dataset to load.
    version : :class:`str`, optional
        Version of the named dataset to load (see available versions in
        documentation). Possibly ``None`` for some datasets.
    reference_genome : :class:`str`, optional
        Reference genome build, ``'GRCh37'`` or ``'GRCh38'``. Possibly ``None``
        for some datasets.
    region : :class:`str`
        Specify region for bucket, ``'us'`` or ``'eu'``, (default is ``'us'``).
    cloud : :class:`str`
        Specify if using Google Cloud Platform or Amazon Web Services,
        ``'gcp'`` or ``'aws'`` (default is ``'gcp'``).

    Note
    ----
    The ``'aws'`` `cloud` platform is currently only available for the ``'us'``
    `region`. If `region` is ``'eu'``, `cloud` must be set to ``'gcp'``.

    Returns
    -------
    :class:`.Table`, :class:`.MatrixTable`, or :class:`.BlockMatrix`
    """

    valid_regions = {'us', 'eu'}
    if region not in valid_regions:
        raise ValueError(f'Specify valid region parameter,'
                         f' received: region={region!r}.\n'
                         f'Valid region values are {valid_regions}.')

    valid_clouds = {'gcp', 'aws'}
    if cloud not in valid_clouds:
        raise ValueError(f'Specify valid cloud parameter,'
                         f' received: cloud={cloud!r}.\n'
                         f'Valid cloud platforms are {valid_clouds}.')

    config_path = pkg_resources.resource_filename(__name__, 'datasets.json')
    assert os.path.exists(config_path), f'{config_path} does not exist'
    with open(config_path) as f:
        datasets = json.load(f)

    names = set([dataset for dataset in datasets])
    if name not in names:
        raise ValueError(f'{name} is not a dataset available in the'
                         f' repository.')

    versions = set(dataset['version'] for dataset in datasets[name]['versions'])
    if version not in versions:
        raise ValueError(f'Version {version!r} not available for dataset'
                         f' {name!r}.\n'
                         f'Available versions: {versions}.')

    reference_genomes = set(dataset['reference_genome']
                            for dataset in datasets[name]['versions'])
    if reference_genome not in reference_genomes:
        raise ValueError(f'Reference genome build {reference_genome!r} not'
                         f' available for dataset {name!r}.\n'
                         f'Available reference genome builds:'
                         f' {reference_genomes}.')

    clouds = set(k for dataset in datasets[name]['versions']
                 for k in dataset['url'].keys())
    if cloud not in clouds:
        raise ValueError(f'Cloud platform {cloud!r} not available for'
                         f' dataset {name}.\n'
                         f'Available platforms: {clouds}.')

    regions = set(k for dataset in datasets[name]['versions']
                  for k in dataset['url'][cloud].keys())
    if region not in regions:
        raise ValueError(f'Region {region!r} not available for dataset'
                         f' {name!r} on cloud platform {cloud!r}.\n'
                         f'Available regions: {regions}.')

    path = [dataset['url'][cloud][region]
            for dataset in datasets[name]['versions']
            if all([dataset['version'] == version,
                    dataset['reference_genome'] == reference_genome])]
    assert len(path) == 1
    path = path[0]
    if path.startswith('s3://'):
        try:
            dataset = _read_dataset(path)
        except hl.utils.java.FatalError:
            dataset = _read_dataset(path.replace('s3://', 's3a://'))
    else:
        dataset = _read_dataset(path)
    return dataset
