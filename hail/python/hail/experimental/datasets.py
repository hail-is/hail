import json
import os
import re
import hail as hl
import pkg_resources


def load_dataset(name,
                 version,
                 reference_genome,
                 region):
    """Load a genetic dataset from Hail's repository.

    Example
    -------

    >>> # Load 1000 Genomes autosomes MatrixTable with GRCh38 coordinates
    >>> mt_1kg = hl.experimental.load_dataset(name='1000_Genomes_autosomes',   # doctest: +SKIP
    ...                                       version='phase_3',
    ...                                       reference_genome='GRCh38',
    ...                                       region='us')

    Parameters
    ----------
    name : :obj:`str`
        Name of the dataset to load.
    version : :obj:`str`
        Version of the named dataset to load
        (see available versions in documentation).
    reference_genome : `GRCh37` or `GRCh38`
        Reference genome build.
    region : `us` or `eu`
        Specify region for GCP bucket.

    Returns
    -------
    :class:`.Table` or :class:`.MatrixTable`"""

    valid_regions = {'us', 'eu'}
    if region not in valid_regions:
        raise ValueError(f'Specify valid region parameter, received: region={region}. '
                         f'Valid regions are {valid_regions}.')

    config_path = pkg_resources.resource_filename(__name__, 'annotation_db.json')
    assert os.path.exists(config_path), f'{config_path} does not exist'
    with open(config_path) as f:
        datasets = json.load(f)

    names = set(dataset for dataset in datasets)
    if name not in names:
        raise ValueError('{} is not a dataset available in the repository.'.format(repr(name)))

    version_ref_genomes = set(x['version'] for x in datasets[name]['versions'])
    versions = set(x.replace("GRCh37", "").replace("GRCh38", "").rstrip("-") for x in version_ref_genomes)
    versions = [None if x is '' else x for x in versions]
    if version not in versions:
        raise ValueError("""Version {0} not available for dataset {1}.
                            Available versions: {{{2}}}.""".format(repr(version),
                                                                   repr(name),
                                                                   repr('","'.join(versions))))
    get_ref_genomes = [re.findall("GRCh\d{2}$", x) for x in version_ref_genomes]
    reference_genomes = set(x[0] for x in get_ref_genomes if x)
    if not reference_genomes:
        reference_genomes = [None]
    if reference_genome not in reference_genomes:
        raise ValueError("""Reference genome build {0} not available for dataset {1}.
                            Available reference genome builds: {{'{2}'}}.""".format(repr(reference_genome),
                                                                                    repr(name),
                                                                                    '\',\''.join(reference_genomes)))

    if version and not reference_genome:
        get_version = version
    elif reference_genome and not version:
        get_version = reference_genome
    else:
        get_version = "-".join([version, reference_genome])

    path = [dataset_version['url'][region]
            for dataset_version in datasets[name]['versions']
            if dataset_version['version'] == get_version][0]
    if path.endswith('.ht'):
        dataset = hl.read_table(path)
    else:
        if not path.endswith('.mt'):
            raise ValueError('Invalid path {}: can only load datasets with .ht or .mt extensions.'.format(repr(path)))
        dataset = hl.read_matrix_table(path)

    return dataset
