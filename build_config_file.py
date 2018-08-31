#!/usr/bin/env python3

import os
import re
import io
import gzip
import json
import requests
import argparse
from subprocess import call, check_output

def fetch_page(url, token=None):
    if token:
        r = requests.get(url, params={'pageToken': token})
    else:
        r = requests.get(url)
    return r.json()

def parse_json(x):
    x = re.sub('Matrix', '', x)
    x = re.sub('Table', '', x)
    x = re.sub('Struct', '', x)
    x = re.sub('Array', '', x)
    x = re.sub(r'\[\[([\w,]*)\],([\w,]*)\]', r'[\1,\2]', x)
    x = re.sub(r'\{', r'{"', x)
    x = re.sub(r'\}', r'"}', x)
    x = re.sub(r'\[', r'["', x)
    x = re.sub(r'\]', r'"]', x)
    x = re.sub(',', '","', x)
    x = re.sub(':', '":"', x)
    x = re.sub('"{', '{', x)
    x = re.sub('}"', '}', x)
    x = re.sub(r'"\[', r'[', x)
    x = re.sub(r'\]"', r']', x)
    x = re.sub('{""}', '""', x)
    x = re.sub(r'"Set\["([\w]+)"\]', r'"Set[\1]"', x)
    x = re.sub(r'"Interval\["([\w()]+)"\]', r'"Interval[\1]"', x)
    x = re.sub(r'\[\[', r'[', x)
    x = re.sub(r'\]\]', r']', x)
    return x

def parse_metadata(stream):
    metadata = json.loads(stream.read().decode('utf-8'))
    n_partitions = len(metadata['components']['partition_counts']['counts'])
    if 'matrix_type' in metadata.keys():
        fields = json.loads(parse_json(metadata['matrix_type']))
    elif 'table_type' in metadata.keys():
        fields = json.loads(parse_json(metadata['table_type']))
    return {'n_partitions': n_partitions, 'fields': fields}

def is_dataset(name):
    return name.endswith('.mt/_SUCCESS') or name.endswith('.ht/_SUCCESS')

def fetch_datasets_gs(url):
    next_page = fetch_page(url)
    tables = [x for x in next_page['items'] if is_dataset(x['name'])]
    for t in tables:
        print('Found dataset {}'.format(t['name'].replace('_SUCCESS', '')))

    while 'nextPageToken' in next_page.keys():
        next_page = fetch_page(url, token=next_page['nextPageToken'])
        new_tables = [x for x in next_page['items'] if is_dataset(x['name'])]
        for t in new_tables:
            print('Found dataset {}'.format(t['name'].replace('_SUCCESS', '')))
        tables.extend(new_tables)

    datasets = []
    for t in tables:
        name = t['name'].split('/')[-2].split('.')[0]
        reference_genome = next(y for y in t['name'].split('/')[-2].split('.') if y.startswith('GRCh'))
        gcs_path = '/'.join(['gs:/', t['bucket'], t['name'].replace('_SUCCESS', '')])
        lifted_over = 'liftover' in t['name']
        stream = gzip.GzipFile(fileobj=io.BytesIO(requests.get(t['selfLink'].replace('_SUCCESS', '') + 'metadata.json.gz', params={'alt': 'media'}).content))
        metadata = parse_metadata(stream)
        datasets.append({
            'name': name,
            'reference_genome': reference_genome,
            'path': gcs_path,
            'n_partitions': metadata['n_partitions'],
            'lifted_over': lifted_over,
            'fields': metadata['fields']
        })

    return datasets

def fetch_datasets_hdfs(root_dir):
    with open(os.devnull, 'w') as f:
        tables = check_output(['hdfs', 'dfs', '-find', root_dir, '-name', '*.{mt,ht}'], stderr=f).decode('utf-8').split()
    datasets = []
    for t in tables:
        print('Found dataset {}'.format(t))
        name = t.split('/')[-1].split('.')[0]
        reference_genome = next(y for y in t.split('/')[-1].split('.') if y.startswith('GRCh'))
        hdfs_path = '/'.join(['hdfs:/', root_dir, t.split('/')[-1]])
        lifted_over = 'liftover' in t
        with open(os.devnull, 'w') as f:
            stream = gzip.GzipFile(fileobj=io.BytesIO(check_output(['hdfs', 'dfs', '-cat', t + '/metadata.json.gz'], stderr=f)))
        metadata = parse_metadata(stream)
        datasets.append({
            'name': name,
            'reference_genome': reference_genome,
            'path': hdfs_path,
            'n_partitions': metadata['n_partitions'],
            'lifted_over': lifted_over,
            'fields': metadata['fields']
        })

    return datasets

def fetch_datasets_local(root_dir):
    tables = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f == '_SUCCESS' and (root.endswith('.mt') or root.endswith('.ht')):
                tables.append(root)
    datasets = []
    for t in tables:
        print('Found dataset {}'.format(t))
        name = t.split('/')[-1].split('.')[0]
        reference_genome = next(y for y in t.split('/')[-1].split('.') if y.startswith('GRCh'))
        path = '/'.join(['file:/', root_dir, t.split('/')[-1]])
        lifted_over = 'liftover' in t
        with gzip.open(t + '/metadata.json.gz', 'r') as f:
            metadata = parse_metadata(f)
        datasets.append({
            'name': name,
            'reference_genome': reference_genome,
            'path': path,
            'n_partitions': metadata['n_partitions'],
            'lifted_over': lifted_over,
            'fields': metadata['fields']
        })

    return datasets

def fetch_datasets_s3(url):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, choices=['local', 'hdfs', 'gs', 's3'], help='Platform the database will be built on (hdfs, gs, or s3).')
    parser.add_argument('--root', default='hail-datasets', help='Root of datasets location (bucket name if gs or s3, file path if hdfs).')
    args = parser.parse_args()

    root = re.sub('^gs:', '', args.root)
    root = re.sub('^s3:', '', args.root)
    root = re.sub('^hdfs:', '', args.root)
    root = re.sub('^file:', '', args.root)
    root = root.strip('/')

    if args.host == 'gs':
        get_url = 'https://www.googleapis.com/storage/v1/b/{}/o'.format(root)
        datasets = fetch_datasets_gs(get_url)
        tmp_file = os.path.join('/tmp', 'hail_datasets.config.json')
        with open(tmp_file, 'w') as f:
            json.dump(datasets, f)
        call(['gsutil', 'cp', tmp_file, 'gs://' + root])

    elif args.host == 'hdfs':
        root_dir = '/' + root
        datasets = fetch_datasets_hdfs(root_dir)
        with open(os.path.join(root_dir, 'hail_datasets.config.json'), 'w') as f:
            json.dump(datasets, f)

    elif args.host == 'local':
        root_dir = '/' + root
        datasets = fetch_datasets_local(root_dir)
        with open(os.path.join(root_dir, 'hail_datasets.config.json'), 'w') as f:
            json.dump(datasets, f)

    elif args.host == 's3':
        pass
