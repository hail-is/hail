#! /usr/bin/python

import argparse

from vep import run, run_vcf_grch37, run_vcf_grch38

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--block-size', type=int, required=True)
parser.add_argument('--data-dir', type=str)
parser.add_argument('--consequence', action='store_true')
parser.add_argument('--tolerate-parse-error', action='store_true')
parser.add_argument('--part-id', type=int, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

# results = run(args.input, args.config, args.block_size, args.data_dir,
#               args.consequence, args.tolerate_parse_error, args.part_id)
#
# with open(args.output, 'w') as out:
#     out.write(f'variant\tvep\tvep_proc_id\n')
#     for v, a, proc_id in results:
#         out.write(f'{v}\t{a}\t{proc_id}\n')

run_vcf_grch37(args.input)
#run_vcf_grch38(args.input)

