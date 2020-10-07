import os
import pkg_resources
import json
import hail as hl


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
    name : :class:`str`
        Name of the dataset to load.
    version : :class:`str`
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

    config_path = pkg_resources.resource_filename(__name__, 'datasets.json')
    assert os.path.exists(config_path), f'{config_path} does not exist'
    with open(config_path) as f:
        datasets = json.load(f)

    names = set([dataset for dataset in datasets])
    if name not in names:
        raise ValueError('{} is not a dataset available in the repository.'.format(repr(name)))

    versions = set(dataset['version'] for dataset in datasets[name]['versions'])
    if version not in versions:
        raise ValueError("""Version {0} not available for dataset {1}.
                            Available versions: {{{2}}}.""".format(repr(version),
                                                                   repr(name),
                                                                   repr('","'.join(versions))))

    reference_genomes = set(dataset['reference_genome'] for dataset in datasets[name]['versions'])
    if reference_genome not in reference_genomes:
        raise ValueError("""Reference genome build {0} not available for dataset {1}.
                            Available reference genome builds: {{'{2}'}}.""".format(repr(reference_genome),
                                                                                    repr(name),
                                                                                    '\',\''.join(reference_genomes)))

    regions = set(k for dataset in datasets[name]['versions'] for k in dataset['url'].keys())
    if region not in regions:
        raise ValueError("""Region {0} not available for dataset {1}.
                            Available regions: {{{2}}}.""".format(repr(region),
                                                                  repr(name),
                                                                  repr('","'.join(regions))))

    path = [dataset['url'][region]
            for dataset in datasets[name]['versions']
            if all([dataset['version'] == version,
                    dataset['reference_genome'] == reference_genome])]
    assert len(path) == 1
    path = path[0]

    if path.endswith('.ht'):
        dataset = hl.read_table(path)
    else:
        if not path.endswith('.mt'):
            raise ValueError('Invalid path {}: can only load datasets with .ht or .mt extensions.'.format(repr(path)))
        dataset = hl.read_matrix_table(path)

    return dataset
