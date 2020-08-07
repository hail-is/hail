import argparse

import hail as hl
from .vcf_combiner import CombinerConfig, run_combiner, parse_sample_mapping


def main():
    parser = argparse.ArgumentParser(description="Driver for Hail's GVCF combiner",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sample_map',
                        help='Path to the sample map, a tab-separated file with two columns. '
                             'The first column is the sample ID, and the second column '
                             'is the GVCF path.')
    parser.add_argument('out_file', help='Path to final combiner output.')
    parser.add_argument('tmp_path', help='Path to folder for intermediate output '
                                         '(should be an object store path, if running on the cloud).')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--genomes', '-G', action='store_true',
                       help='Indicates that the combiner is operating on genomes. '
                            'Affects how the genome is partitioned on input.')
    group.add_argument('--exomes', '-E', action='store_true',
                       help='Indicates that the combiner is operating on exomes. '
                            'Affects how the genome is partitioned on input.')
    group.add_argument('--import-interval-size', type=int,
                       help='Interval size for partitioning the reference genome for GVCF import.')
    parser.add_argument('--log', help='Hail log path.')
    parser.add_argument('--header',
                        help='External header, must be readable by all executors. '
                             'WARNING: if this option is used, the sample names in the '
                             'GVCFs will be overridden by the names in sample map.',
                        required=False)
    parser.add_argument('--branch-factor', type=int, default=CombinerConfig.default_branch_factor, help='Branch factor.')
    parser.add_argument('--batch-size', type=int, default=CombinerConfig.default_batch_size, help='Batch size.')
    parser.add_argument('--target-records', type=int, default=CombinerConfig.default_target_records, help='Target records per partition.')
    parser.add_argument('--overwrite', help='overwrite the output path', action='store_true')
    parser.add_argument('--key-by-locus-and-alleles', help='Key by both locus and alleles in the final output.', action='store_true')
    parser.add_argument('--reference-genome', default='GRCh38', help='Reference genome.')
    args = parser.parse_args()
    hl.init(log=args.log)

    if not args.overwrite and hl.utils.hadoop_exists(args.out_file):
        raise FileExistsError(f"path '{args.out_file}' already exists, use --overwrite to overwrite this path")

    sample_names, sample_paths = parse_sample_mapping(args.sample_map)
    run_combiner(sample_paths,
                 args.out_file,
                 args.tmp_path,
                 header=args.header,
                 sample_names=sample_names,
                 batch_size=args.batch_size,
                 branch_factor=args.branch_factor,
                 target_records=args.target_records,
                 import_interval_size=args.import_interval_size,
                 use_genome_default_intervals=args.genomes,
                 use_exome_default_intervals=args.exomes,
                 overwrite=args.overwrite,
                 reference_genome=args.reference_genome,
                 key_by_locus_and_alleles=args.key_by_locus_and_alleles)


if __name__ == '__main__':
    main()
