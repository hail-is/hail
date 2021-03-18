import os

import hail as hl
from hail.experimental.vcf_combiner import vcf_combiner as vc
from hail.utils.java import Env
from hail.utils.misc import new_temp_file
from ..helpers import resource, startTestHailContext, stopTestHailContext, fails_local_backend, fails_service_backend

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


all_samples = ['HG00308', 'HG00592', 'HG02230', 'NA18534', 'NA20760',
               'NA18530', 'HG03805', 'HG02223', 'HG00637', 'NA12249',
               'HG02224', 'NA21099', 'NA11830', 'HG01378', 'HG00187',
               'HG01356', 'HG02188', 'NA20769', 'HG00190', 'NA18618',
               'NA18507', 'HG03363', 'NA21123', 'HG03088', 'NA21122',
               'HG00373', 'HG01058', 'HG00524', 'NA18969', 'HG03833',
               'HG04158', 'HG03578', 'HG00339', 'HG00313', 'NA20317',
               'HG00553', 'HG01357', 'NA19747', 'NA18609', 'HG01377',
               'NA19456', 'HG00590', 'HG01383', 'HG00320', 'HG04001',
               'NA20796', 'HG00323', 'HG01384', 'NA18613', 'NA20802']


@fails_service_backend()
@fails_local_backend()
def test_1kg_chr22():
    out_file = new_temp_file(extension='mt')

    sample_names = all_samples[:5]
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', f'{s}.hg38.g.vcf.gz') for s in sample_names]
    vc.run_combiner(paths,
                    out_file=out_file,
                    tmp_path=Env.hc()._tmpdir,
                    branch_factor=2,
                    batch_size=2,
                    reference_genome='GRCh38',
                    use_exome_default_intervals=True)

    sample_data = dict()
    for sample, path in zip(sample_names, paths):
        ht = hl.import_vcf(path, force_bgz=True, reference_genome='GRCh38').localize_entries('entries')
        n, n_variant = ht.aggregate((hl.agg.count(), hl.agg.count_where(ht.entries[0].GT.is_non_ref())))
        sample_data[sample] = (n, n_variant)

    mt = hl.read_matrix_table(out_file)
    mt = mt.annotate_cols(n=hl.agg.count(), n_variant=hl.agg.count_where(
        mt.LGT.is_non_ref()))  # annotate the number of non-missing records

    combined_results = hl.tuple([mt.s, mt.n, mt.n_variant]).collect()
    assert len(combined_results) == len(sample_names)

    for sample, n, n_variant in combined_results:
        true_n, true_n_variant = sample_data[sample]
        assert n == true_n, sample
        assert n_variant == true_n_variant, sample

def default_exome_intervals(rg):
    return vc.calculate_even_genome_partitioning(rg, 2 ** 32)  # 4 billion, larger than any contig

@fails_service_backend()
@fails_local_backend()
def test_gvcf_1k_same_as_import_vcf():
    path = os.path.join(resource('gvcfs'), '1kg_chr22', f'HG00308.hg38.g.vcf.gz')
    [mt] = hl.import_gvcfs([path], default_exome_intervals('GRCh38'), reference_genome='GRCh38')
    assert mt._same(hl.import_vcf(path, force_bgz=True, reference_genome='GRCh38').key_rows_by('locus'))

@fails_service_backend()
@fails_local_backend()
def test_gvcf_subset_same_as_import_vcf():
    path = os.path.join(resource('gvcfs'), 'subset', f'HG00187.hg38.g.vcf.gz')
    [mt] = hl.import_gvcfs([path], default_exome_intervals('GRCh38'), reference_genome='GRCh38')
    assert mt._same(hl.import_vcf(path, force_bgz=True, reference_genome='GRCh38').key_rows_by('locus'))

@fails_service_backend()
@fails_local_backend()
def test_key_by_locus_alleles():
    out_file = new_temp_file(extension='mt')

    sample_names = all_samples[:5]
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', f'{s}.hg38.g.vcf.gz') for s in sample_names]
    vc.run_combiner(paths,
                    out_file=out_file,
                    tmp_path=Env.hc()._tmpdir,
                    reference_genome='GRCh38',
                    key_by_locus_and_alleles=True,
                    use_exome_default_intervals=True)

    mt = hl.read_matrix_table(out_file)
    assert(list(mt.row_key) == ['locus', 'alleles'])
    mt._force_count_rows()


@fails_service_backend()
@fails_local_backend()
def test_non_ref_alleles_set_to_missing():
    path = os.path.join(resource('gvcfs'), 'non_ref_call.g.vcf.gz')
    out_file = new_temp_file(extension='mt')
    vc.run_combiner([path, path],
                    out_file=out_file,
                    tmp_path=Env.hc()._tmpdir,
                    branch_factor=2,
                    batch_size=2,
                    reference_genome='GRCh38',
                    use_exome_default_intervals=True)

    mt = hl.read_matrix_table(out_file)
    n_alleles = hl.len(mt.alleles)
    gt_idx = hl.experimental.lgt_to_gt(mt.LGT, mt.LA).unphased_diploid_gt_index()
    assert mt.aggregate_entries(
        hl.agg.all(gt_idx < (n_alleles * (n_alleles + 1)) / 2))

@fails_service_backend()
@fails_local_backend()
def test_contig_recoding():
    path1 = os.path.join(resource('gvcfs'), 'recoding', 'HG00187.hg38.g.vcf.gz')
    path2 = os.path.join(resource('gvcfs'), 'recoding', 'HG00187.hg38.recoded.g.vcf.gz')

    out_file_1 = new_temp_file(extension='mt')
    out_file_2 = new_temp_file(extension='mt')

    vc.run_combiner([path1, path1], out_file_1,
                    Env.hc()._tmpdir,
                    reference_genome='GRCh38',
                    use_exome_default_intervals=True)
    vc.run_combiner([path2, path2], out_file_2,
                    Env.hc()._tmpdir,
                    reference_genome='GRCh38',
                    contig_recoding={'22': 'chr22'},
                    use_exome_default_intervals=True)

    mt1 = hl.read_matrix_table(out_file_1)
    mt2 = hl.read_matrix_table(out_file_2)

    assert mt1.count() == mt2.count()
    assert mt1._same(mt2)

@fails_service_backend()
@fails_local_backend()
def test_sample_override():
    out_file = new_temp_file(extension='mt')

    sample_names = all_samples[:5]
    new_names = [f'S{i}' for i, _ in enumerate(sample_names)]
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', f'{s}.hg38.g.vcf.gz') for s in sample_names]
    header_path = paths[0]
    vc.run_combiner(paths,
                    out_file=out_file,
                    tmp_path=Env.hc()._tmpdir,
                    reference_genome='GRCh38',
                    header=header_path,
                    sample_names=new_names,
                    key_by_locus_and_alleles=True,
                    use_exome_default_intervals=True)
    mt_cols = hl.read_matrix_table(out_file).key_cols_by().cols()
    mt_names = mt_cols.aggregate(hl.agg.collect(mt_cols.s))
    assert new_names == mt_names
