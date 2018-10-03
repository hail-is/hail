
import sys
import json
import hail as hl
from io import StringIO

def table_bars(*widths):
    return ' '.join(['=' * w for w in widths])

def table_row(*cols):
    return ' '.join([c[0] + ' '*(c[1] - len(c[0])) for c in cols])

class HailDataset(object):

    def __init__(self, path):

        # load dataset
        self.path = path.strip('/')
        if self.path.endswith('.ht'):
            self.type = 'Table'
            self.dataset = hl.read_table(self.path)
        else:
            self.type = 'MatrixTable'
            self.dataset = hl.read_matrix_table(self.path)

        # extract schema
        stdout = sys.stdout
        schema = StringIO()
        sys.stdout = schema
        self.dataset.describe()
        sys.stdout = stdout
        self.schema = schema.getvalue()

        # extract metadata
        try:
            self.metadata = self.dataset['metadata'].collect()[0]
        except LookupError:
            self.metadata = None

        # extract metadata fields,
        # if metadata global field doesn't exist, or if it doesn't contain at 
        # least name, version, and reference genome fields, dataset is invalid
        # and will not be registered in the database
        if self.metadata:
            try:
                self.name = self.metadata['name']
                self.version = self.metadata['version']
                self.reference_genome = self.metadata['reference_genome']
                self.n_rows = self.metadata['n_rows'] if 'n_rows' in self.metadata else None
                self.n_cols = self.metadata['n_cols'] if 'n_cols' in self.metadata else None
                self.n_partitions = self.metadata['n_partitions'] if 'n_partitions' in self.metadata else none
            except LookupError:
                self.is_valid = False
            else:
                self.is_valid = True
        else:
            self.is_valid = False

def fetch_datasets(paths):
    datasets = []
    for path in paths:
        dataset = HailDataset(path)
        if dataset.is_valid:
            datasets.append({
                'path': dataset.path,
                'name': dataset.name,
                'version': dataset.version,
                'reference_genome': dataset.reference_genome,
                'type': dataset.type,
                'n_rows': dataset.n_rows,
                'n_cols': dataset.n_cols,
                'n_partitions': dataset.n_partitions,
                'schema': dataset.schema
            })
    return datasets

if __name__ == '__main__':

    bucket = 'gs://hail-datasets/'
    paths = [x['path'] for x in hl.hadoop_ls(bucket) if x['path'].endswith('.ht') or x['path'].endswith('.mt')]
    config = []
    datasets = fetch_datasets(paths)

    with hl.hadoop_open(bucket + 'datasets.json', 'w') as f:
        json.dump(datasets, f)

    dataset_names = sorted(list(set([d['name'] for d in datasets])))

    doc_rows = []
    for name in dataset_names:
        versions = sorted(list(set([d['version'] for d in datasets if d['name']==name])))
        reference_genomes = sorted(list(set([d['reference_genome'] for d in datasets if d['name']==name])))
        doc_rows.append((':ref:`' + name + '`', ', '.join(versions), ', '.join(reference_genomes)))

    header = ['Name', 'Versions', 'Reference Genomes']
    
    col0_width = max([len(dr[0]) for dr in doc_rows] + [len(header[0])])
    col1_width = max([len(dr[1]) for dr in doc_rows] + [len(header[1])])
    col2_width = max([len(dr[2]) for dr in doc_rows] + [len(header[2])]) 

    datasets_rst = ['.. _sec-datasets:',
                    '',
                    ':tocdepth: 1',
                    '',
                    '========',
                    'Datasets',
                    '========',
                    '',
                    '.. warning:: ',
                    '    All functionality described on this page is experimental.',
                    '    Datasets and method are subject to change.',
                    '',
                    'This page describes genetic datasets that are hosted in a public repository',
                    'on Google Cloud Platform and are available for use through Hail\'s',
                    ':meth:`.load_dataset` function.',
                    '',
                    'To load a dataset from this repository into a Hail pipeline, provide the name,',
                    'version, and reference genome build of the dataset you would like to use as',
                    'strings to the :meth:`.load_dataset` function. The available dataset names,',
                    'versions, and reference genome builds are listed in the table below.',
                    '',
                    table_bars(col0_width, col1_width, col2_width),
                    table_row((header[0], col0_width), (header[1], col1_width), (header[2], col2_width)),
                    table_bars(col0_width, col1_width, col2_width)] + [
                    table_row((dr[0], col0_width), (dr[1], col1_width), (dr[2], col2_width)) for dr in doc_rows] + [
                    table_bars(col0_width, col1_width, col2_width),
                    '',
                    '.. toctree::',
                    '    :hidden:',
                    ''] + [
                    '    datasets/{}.rst'.format(dr[0].replace(':ref:', '').replace('`', '')) for dr in doc_rows] + [
                    '']

    with hl.hadoop_open(bucket + 'docs/datasets.rst', 'w') as f:
        f.write('\n'.join(datasets_rst))

    for name in dataset_names:
        versions = sorted(list(set([d['version'] for d in datasets if d['name']==name])))
        reference_genomes = sorted(list(set([d['reference_genome'] for d in datasets if d['name']==name])))
        dataset_type = [d['type'] for d in datasets if d['name']==name][0]
        schema = [(d['schema'], d['version'], d['reference_genome']) for d in datasets if d['name']==name][0]
        with hl.hadoop_open(bucket + 'docs/datasets/{}.rst'.format(name), 'w') as f:
            f.write('\n'.join([
                '.. _{}:'.format(name),
                '',
                name,
                '=' * len(name),
                '',
                '*  **Versions:** {}'.format(', '.join(versions)),
                '*  **Reference genome builds:** {}'.format(', '.join(reference_genomes)),
                '*  **Type:** {}'.format(dataset_type),
                '',
                'Schema ({0}, {1})'.format(schema[1], schema[2]),
                '~'*(len('Schema ({0}, {1})'.format(schema[1], schema[2]))),
                '',
                '.. code-block:: text',
                '',
                '\n'.join(['    ' + line for line in schema[0].split('\n')]),
                ''
            ]))
