#!/usr/bin/env python3

import requests, h5py
import numpy as np
from subprocess import call

print('Fetching "ica_cord_blood_h5.h5"...')
response = requests.get('https://s3.amazonaws.com/preview-ica-expression-data/ica_cord_blood_h5.h5')
with open('/tmp/ica_cord_blood_h5.h5', 'wb') as f:
    f.write(response.content)

h5 = h5py.File('/tmp/ica_cord_blood_h5.h5', 'r')['GRCh38']

print('Extracting data...')
np.savetxt('/tmp/Human_Cell_Atlas_cord_blood_immunocytes.counts.tsv',
           np.column_stack((h5['indices'], h5['data'])),
           delimiter='\t',
           header='gene_idx\tcount',
           comments='',
           fmt='%s')

np.savetxt('/tmp/Human_Cell_Atlas_cord_blood_immunocytes.barcodes.tsv',
           np.column_stack((h5['indptr'][:-1], h5['barcodes'])),
           delimiter='\t',
           header='barcode_idx\tbarcode',
           comments='',
           fmt='%s')

np.savetxt('/tmp/Human_Cell_Atlas_cord_blood_immunocytes.genes.tsv',
           np.column_stack((h5['genes'], h5['gene_names'])),
           delimiter='\t',
           header='gene_id\tgene_name',
           comments='',
           fmt='%s')

print('Block compressing...')
call(['bgzip', '/tmp/Human_Cell_Atlas_cord_blood_immunocytes.counts.tsv'])
call(['bgzip', '/tmp/Human_Cell_Atlas_cord_blood_immunocytes.barcodes.tsv'])
call(['bgzip', '/tmp/Human_Cell_Atlas_cord_blood_immunocytes.genes.tsv'])

print('Copying block compressed files...')
call(['gsutil', 'cp', '/tmp/Human_Cell_Atlas_cord_blood_immunocytes.counts.tsv.gz',
      'gs://hail-datasets-raw-data/Human_Cell_Atlas/Human_Cell_Atlas_cord_blood_immunocytes/counts.tsv.bgz'])
call(['gsutil', 'cp', '/tmp/Human_Cell_Atlas_cord_blood_immunocytes.barcodes.tsv.gz',
      'gs://hail-datasets-raw-data/Human_Cell_Atlas/Human_Cell_Atlas_cord_blood_immunocytes/barcodes.tsv.bgz'])
call(['gsutil', 'cp', '/tmp/Human_Cell_Atlas_cord_blood_immunocytes.genes.tsv.gz',
      'gs://hail-datasets-raw-data/Human_Cell_Atlas/Human_Cell_Atlas_cord_blood_immunocytes/genes.tsv.bgz'])

