import os.path
from tempfile import TemporaryDirectory

import hail as hl
from hail.vds.combiner import combine_variant_datasets, new_combiner, transform_gvcf
from hail.vds.combiner.combine import (
    combine_gvcfs,
    calculate_even_genome_partitioning,
)

from .resources import empty_gvcf, single_gvcf, chr22_gvcfs
from .utils import benchmark

COMBINE_GVCF_MAX = 100
MAX_TO_COMBINE = 20 * COMBINE_GVCF_MAX


def chunks(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def setup(path):
    return hl.import_vcf(path, reference_genome='GRCh38', force=True)


@benchmark(args=empty_gvcf.handle())
def compile_2k_merge(path):
    flagname = 'no_ir_logging'
    prev_flag_value = hl._get_flags(flagname).get(flagname)
    try:
        hl._set_flags(**{flagname: '1'})
        vcf = setup(path)
        vcfs = [transform_gvcf(vcf, [])] * COMBINE_GVCF_MAX
        combined = [combine_variant_datasets(vcfs)] * 20
        with TemporaryDirectory() as tmpdir:
            hl.vds.write_variant_datasets(combined, os.path.join(tmpdir, 'combiner-multi-write'), overwrite=True)
    finally:
        hl._set_flags(**{flagname: prev_flag_value})


@benchmark(args=empty_gvcf.handle())
def python_only_10k_transform(path):
    vcf = setup(path)
    vcfs = [vcf] * 10_000
    _ = [transform_gvcf(vcf, []) for vcf in vcfs]


@benchmark(args=empty_gvcf.handle())
def python_only_10k_combine(path):
    vcf = setup(path)
    mt = transform_gvcf(vcf, [])
    mts = [mt] * 10_000
    _ = [combine_variant_datasets(mts) for mts in chunks(mts, COMBINE_GVCF_MAX)]


@benchmark(args=single_gvcf.handle())
def import_and_transform_gvcf(path):
    mt = setup(path)
    vds = transform_gvcf(mt, [])
    vds.reference_data._force_count_rows()
    vds.variant_data._force_count_rows()


@benchmark(args=single_gvcf.handle())
def import_gvcf_force_count(path):
    mt = setup(path)
    mt._force_count_rows()


@benchmark(args=[chr22_gvcfs.handle(name) for name in chr22_gvcfs.samples])
def vds_combiner_chr22(*paths):
    with TemporaryDirectory() as tmpdir:
        with TemporaryDirectory() as outpath:
            parts = hl.eval([hl.parse_locus_interval('chr22:start-end', reference_genome='GRCh38')])

            combiner = new_combiner(
                output_path=outpath,
                intervals=parts,
                temp_path=tmpdir,
                gvcf_paths=list(paths),
                reference_genome='GRCh38',
                branch_factor=16,
                target_records=10000000,
            )
            combiner.run()
