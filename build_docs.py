
import json
import hail as hl

with hl.hadoop_open('gs://hail-datasets/hail_datasets.config.json', 'r') as f:
    config = json.load(f)

datasets_builds_type = {}
for dataset in config:
    name = dataset['name']
    if name in datasets_builds_type:
        datasets_builds_type[name][0].append(dataset['reference_genome'])
    else:
        dataset_type = dataset['path'].strip('/')[-2:]
        datasets_builds_type[name] = [[dataset['reference_genome']], dataset_type]

for k, v in datasets_builds_type.items():
    print(k, v)

header = ['Dataset name', 'Reference builds available', 'Type']
col0 = max([len(':ref:`{}`'.format(x)) for x in datasets_builds_type.keys()]) + 1
col1 = len(header[1])
col2 = len(':class:`.MatrixTable`')

datasets_rst = [
    '.. _sec-datasets:',
    '',
    ':tocdepth: 1',
    '',
    'This page describes genetic datasets that have already been imported into',
    'Hail-friendly formats and are available for use through Hail\'s',
    ':meth:`.load_dataset` function.',
    '',
    '========',
    'Datasets',
    '========',
    '',
    '=' * col0 + ' ' + '=' * col1 + ' ' + '=' * col2,
    header[0] + ' ' * (col0 - len(header[0]) + 1) + header[1] + ' ' + header[2],
    '=' * col0 + ' ' + '=' * col1 + ' ' + '=' * col2
]

entries = []
for k, v in datasets_builds_type.items():
    builds = ', '.join(v[0])
    dataset_type = ':class:`.MatrixTable`' if v[1] == 'mt' else ':class:`.Table`'
    entries.append(':ref:`{}`'.format(k) + ' ' * (col0 - len(k) + 1) + builds + ' ' * (col1 - len(builds) + 1) + dataset_type)

datasets_rst += entries
datasets_rst.append('=' * col0 + ' ' + '=' * col1 + ' ' + '=' * col2)
datasets_rst.extend([
    '',
    '.. toctree::',
    '    :glob:',
    '    :hidden:',
    '',
    '    datasets/*',
    ''
])

with hl.hadoop_open('gs://hail-datasets/docs/datasets.rst', 'w') as f:
    f.write('\n'.join(datasets_rst))

for k, v in datasets_builds_type.items():
    with hl.hadoop_open('gs://hail-datasets/docs/datasets/{}.rst'.format(k), 'w') as f:
        f.write('\n'.join([
            '.. _{}:'.format(k),
            '',
            k,
            '=' * len(k),
            ''
        ]))
