#! /usr/bin/python

import argparse

from vep import get_csq_header

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, required=True)
parser.add_argument('--data-dir', type=str)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

csq_header = get_csq_header(args.config, args.data_dir)
with open(args.output, 'w') as out:
    out.write(f'{csq_header}\n')
