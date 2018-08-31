
import json
import hail as hl


def load_dataset(dataset_name, reference_genome, config_file='gs://hail-datasets/hail-data/hail_datasets.config.json'):
    """Load a Hail-formatted genetic dataset.
        
       Datasets currently available:
       - """

    with hl.hadoop_open(config_file, 'r') as f:
        config = json.load(f)

    path = next(x['path'] for x in config if x['name'] == dataset_name and x['reference_genome'] == reference_genome)

    if path.endswith('.ht'):
        dataset = hl.read_table(path)
    elif path.endswith('.mt'):
        dataset = hl.read_matrix_table(path)

    return dataset
