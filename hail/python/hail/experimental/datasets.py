
import json
import hail as hl


def load_dataset(dataset_name,
                 reference_genome,
                 config_file='gs://hail-datasets/hail_datasets.config.json'):
    """Load a Hail-formatted genetic dataset.

    Example
    -------

    >>> # Load 1000 Genomes chromosome X MatrixTable with GRCh38 coordinates
    >>> mt_1kg = hl.experimental.load_dataset('1000_genomes_phase3_chrX',   # doctest: +SKIP
    ...                                       reference_genome='GRCh38')

    Parameters
    ----------
    dataset_name : :obj:`str`
        Name of the dataset to load.
    reference_genome : `GRCh37` or `GRCh38`
        Reference genome build.
    config_file : :obj:`str`, optional
        Path of the datasets configuration file.
        Leave as default if running on Google Cloud Platform.

    Returns
    -------
    :class:`.Table` or :class:`.MatrixTable`"""

    with hl.hadoop_open(config_file, 'r') as f:
        config = json.load(f)

    builds = [{'path': x['path'], 'reference_genome': x['reference_genome']}
              for x in config if x['name'] == dataset_name]

    if not builds:
        raise KeyError("Dataset '{}' not found.".format(dataset_name))

    path = [x['path'] for x in builds
            if x['reference_genome'] == reference_genome]
    if not path:
        raise ValueError("""Reference genome '{0}' not available for dataset '{1}'. Available reference genomes: '{2}'.""".format(
                         reference_genome,
                         dataset_name,
                         ', '.join([x['reference_genome'] for x in builds])))
    else:
        path = path[0].strip('/')

    if path.endswith('.ht'):
        dataset = hl.read_table(path)
    else:
        assert path.endswith('.mt')
        dataset = hl.read_matrix_table(path)

    return dataset
