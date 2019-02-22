#!/usr/bin/env python3

import requests, h5py
import numpy as np
import pandas as pd
from subprocess import call

URL_ROOT = 'https://s3.amazonaws.com/preview-ica-expression-data'

print('Fetching "ica_cord_blood_h5.h5"...')
response = requests.get(f'{URL_ROOT}/ica_cord_blood_h5.h5')
with open('/tmp/ica_cord_blood_h5.h5', 'wb') as f:
    f.write(response.content)

f = h5py.File('/tmp/ica_cord_blood_h5.h5', 'r')

with open('/tmp/ica_cord_blood.header.tsv', 'w') as f_out:
    f_out.write('\t'.join(['barcode'] + [x.decode('utf-8') for x in f['GRCh38']['genes']]) + '\n')

n_genes = f['GRCh38']['shape'][0]
n_cells = f['GRCh38']['shape'][1]

chunk_size = 10000
n_chunks = int(np.ceil(n_cells / chunk_size))

chunk_start = 0
chunk_end = chunk_start + chunk_size

for chunk in range(n_chunks):
    print(f'Writing chunk {chunk}...')
    counts_array = np.zeros([chunk_size, n_genes], dtype=int)
    breaks = f['GRCh38']['indptr'][chunk_start:(chunk_end + 1)]
    row_bounds = zip(breaks[:-1], breaks[1:])
    for i, (start, end) in enumerate(row_bounds):
        col_idxs = f['GRCh38']['indices'][start:end]
        counts = f['GRCh38']['data'][start:end]
        counts_array[i % chunk_size, col_idxs] = counts
    df = pd.DataFrame(data=counts_array, index=f['GRCh38']['barcodes'][chunk_start:chunk_end], columns=None)
    df.to_csv(f'/tmp/ica_cord_blood.chunk_{chunk}.tsv', sep='\t', float_format='%i', index=True, header=False)
    chunk_start = i + 1
    chunk_end = min(i + 1 + chunk_size, n_cells + 1)

print('Concatenating chunks...')
with open('/tmp/ica_cord_blood.tsv', 'w') as f_out:
    call(['cat', '/tmp/ica_cord_blood.header.tsv', '/tmp/ica_cord_blood.chunk_*.tsv'], stdout=f_out)

print('Block compressing concatenated file...')
call(['bgzip', '/tmp/ica_cord_blood.tsv'])

print('Copying block compressed file...')
call(['gsutil', 'cp', '/tmp/ica_cord_blood.tsv.gz',
      'gs://hail-datasets-raw-data/Human_Cell_Atlas/Human_Cell_Atlas_immune_cells_cord_blood_raw_counts.tsv.bgz'])

print('Removing temporary files...')
call(['rm', '/tmp/ica_cord_blood*'])

