import os
import pytest

import hail as hl
from hail.utils import new_temp_file
from hail.vds.combiner.combine import defined_entry_fields
from ..helpers import startTestHailContext, stopTestHailContext, resource, fails_local_backend, fails_service_backend

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


# run this method to regenerate the combined VDS from 5 samples
def generate_5_sample_vds():
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', path) for path in ['HG00187.hg38.g.vcf.gz',
                                                                             'HG00190.hg38.g.vcf.gz',
                                                                             'HG00308.hg38.g.vcf.gz',
                                                                             'HG00313.hg38.g.vcf.gz',
                                                                             'HG00320.hg38.g.vcf.gz']]
    parts = [
        hl.Interval(start=hl.Struct(locus=hl.Locus('chr22', 1, reference_genome='GRCh38')),
                    end=hl.Struct(locus=hl.Locus('chr22', hl.get_reference('GRCh38').contig_length('chr22') - 1,
                                                 reference_genome='GRCh38')),
                    includes_end=True)
    ]
    vcfs = hl.import_gvcfs(paths, parts, reference_genome='GRCh38', array_elements_required=False)
    to_keep = defined_entry_fields(vcfs[0].filter_rows(hl.is_defined(vcfs[0].info.END)), 100_000)
    vds = hl.vds.combiner.combine_variant_datasets([hl.vds.combiner.transform_gvcf(mt, to_keep) for mt in vcfs])
    vds.variant_data = vds.variant_data._key_rows_by_assert_sorted('locus', 'alleles')
    vds.write(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'), overwrite=True)


def test_validate():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    vds.validate()

    with pytest.raises(ValueError):
        hl.vds.VariantDataset(
            vds.reference_data.annotate_rows(arr=[0, 1]).explode_rows('arr'),
            vds.variant_data).validate()

    with pytest.raises(ValueError):
        hl.vds.VariantDataset(
            vds.reference_data.annotate_entries(
                END=hl.or_missing(vds.reference_data.locus.position % 2 == 0, vds.reference_data.END)),
            vds.variant_data).validate()


@fails_local_backend()
@fails_service_backend()
def test_multi_write():
    vds1 = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    to_keep = vds1.variant_data.filter_cols(vds1.variant_data.s == 'HG00187').cols()
    vds2 = hl.vds.filter_samples(vds1, to_keep)

    path1 = new_temp_file()
    path2 = new_temp_file()
    hl.vds.write_variant_datasets([vds1, vds2], [path1, path2])

    assert hl.vds.read_vds(path1)._same(vds1)
    assert hl.vds.read_vds(path2)._same(vds2)

@fails_local_backend
@fails_service_backend
def test_conversion_equivalence():
    gvcfs = [os.path.join(resource('gvcfs'), '1kg_chr22', path) for path in ['HG00187.hg38.g.vcf.gz',
                                                                             'HG00190.hg38.g.vcf.gz',
                                                                             'HG00308.hg38.g.vcf.gz',
                                                                             'HG00313.hg38.g.vcf.gz',
                                                                             'HG00320.hg38.g.vcf.gz']]

    tmpdir = new_temp_file()
    mt_path = new_temp_file()
    vds_path = new_temp_file()

    hl.experimental.run_combiner(gvcfs, mt_path, tmpdir, use_exome_default_intervals=True, reference_genome='GRCh38',
                                 overwrite=True, intervals=[hl.eval(hl.parse_locus_interval('chr22', 'GRCh38'))],
                                 key_by_locus_and_alleles=True)

    svcr = hl.read_matrix_table(mt_path)

    vds = hl.vds.VariantDataset.from_merged_representation(svcr).checkpoint(vds_path)
    ref = vds.reference_data
    var = vds.variant_data

    assert svcr.aggregate_entries(hl.agg.count_where(hl.is_defined(svcr.END))) == ref.aggregate_entries(hl.agg.count())
    assert svcr.aggregate_entries(hl.agg.count()) == ref.aggregate_entries(hl.agg.count()) + var.aggregate_entries(
        hl.agg.count())

    svcr_readback = hl.vds.to_merged_sparse_mt(vds)

    assert svcr._same(svcr_readback)


@fails_local_backend
@fails_service_backend
def test_sampleqc_old_new_equivalence():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    sqc = hl.vds.sample_qc(vds)

    dense = hl.vds.to_dense_mt(vds)
    dense = dense.transmute_entries(GT=hl.vds.lgt_to_gt(dense.LGT, dense.LA))
    res = hl.sample_qc(dense)

    res = res.annotate_cols(sample_qc_new=sqc[res.s])

    fields_to_test = [
        'n_het',
        'n_hom_var',
        'n_non_ref',
        'n_singleton',
        'n_snp',
        'n_insertion',
        'n_deletion',
        'n_transition',
        'n_transversion',
        'n_star',
        'r_ti_tv',
        'r_het_hom_var',
        'r_insertion_deletion'
    ]

    res.sample_qc.describe()
    sqc.describe()
    assert res.aggregate_cols(hl.all(
        *(hl.agg.all(res.sample_qc[field] == res.sample_qc_new[field]) for field in fields_to_test)
    ))


def test_filter_samples_and_merge():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))

    samples = vds.variant_data.cols()
    samples = samples.add_index()

    samples1 = samples.filter(samples.idx < 2)
    samples2 = samples.filter(samples.idx >= 2)

    split1 = hl.vds.filter_samples(vds, samples1, remove_dead_alleles=True)

    assert split1.variant_data.count_cols() == 2
    assert split1.reference_data.count_cols() == 2

    split2 = hl.vds.filter_samples(vds, samples2, remove_dead_alleles=True)

    assert split2.variant_data.count_cols() == 3
    assert split2.reference_data.count_cols() == 3

    merged = hl.vds.combiner.combine_variant_datasets([split1, split2])

    assert merged.reference_data._same(vds.reference_data)
    assert merged.variant_data._same(vds.variant_data)


@fails_local_backend
@fails_service_backend
def test_segment_intervals():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))

    contig_len = vds.reference_data.locus.dtype.reference_genome.lengths['chr22']
    breakpoints = hl.literal([*range(1, contig_len, 5_000_000), contig_len])
    intervals = hl.range(hl.len(breakpoints) - 1) \
        .map(lambda i: hl.struct(
        interval=hl.locus_interval('chr22', breakpoints[i], breakpoints[i + 1], reference_genome='GRCh38')))
    intervals_ht = hl.Table.parallelize(intervals, key='interval')

    path = new_temp_file()
    r = hl.vds.segment_reference_blocks(vds.reference_data, intervals_ht)
    r.write(path)
    after = hl.read_matrix_table(path)

    es = after.entries()
    es = es.filter((es.END < es.locus.position) | (es.END >= es.interval.end.position))
    if es.count() > 0:
        es.show(width=1000)
        assert False, "found entries with END < position or END >= interval end"

    before = vds.reference_data

    sum_per_sample_before = before.select_cols(
        ref_block_bases=hl.agg.sum(before.END + 1 - before.locus.position)).cols()
    sum_per_sample_after = after.select_cols(ref_block_bases=hl.agg.sum(after.END + 1 - after.locus.position)).cols()

    before_coverage = sum_per_sample_before.collect()
    after_coverage = sum_per_sample_after.collect()
    assert before_coverage == after_coverage


@fails_local_backend
@fails_service_backend
def test_interval_coverage():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))

    interval1 = 'chr22:10678825-10678835'
    interval2 = 'chr22:10678970-10678979'

    intervals = hl.Table.parallelize(
        list(hl.struct(interval=hl.parse_locus_interval(x, reference_genome='GRCh38')) for x in [interval1, interval2]),
        key='interval')

    checkpoint_path = new_temp_file()
    r = hl.vds.interval_coverage(vds, intervals).checkpoint(checkpoint_path)
    assert r.aggregate_rows(hl.agg.collect((hl.format('%s:%d-%d', r.interval.start.contig, r.interval.start.position,
                                                      r.interval.end.position), r.interval_size))) == [(interval1, 10),
                                                                                                       (interval2, 9)]

    r.entries().show(width=10000, n_rows=10000)

    observed = r.aggregate_entries(hl.agg.collect(r.entry))
    expected = [
        hl.Struct(bases_over_gq_threshold=(10, 0), sum_dp=55, fraction_over_gq_threshold=(1.0, 0.0), mean_dp=5.5),
        hl.Struct(bases_over_gq_threshold=(10, 0), sum_dp=45, fraction_over_gq_threshold=(1.0, 0.0), mean_dp=4.5),
        hl.Struct(bases_over_gq_threshold=(0, 0), sum_dp=0, fraction_over_gq_threshold=(0.0, 0.0), mean_dp=0),
        hl.Struct(bases_over_gq_threshold=(10, 0), sum_dp=30, fraction_over_gq_threshold=(1.0, 0.0), mean_dp=3.0),
        hl.Struct(bases_over_gq_threshold=(9, 0), sum_dp=10, fraction_over_gq_threshold=(0.9, 0.0), mean_dp=1.0),

        hl.Struct(bases_over_gq_threshold=(9, 9), sum_dp=153, fraction_over_gq_threshold=(1.0, 1.0), mean_dp=17.0),
        hl.Struct(bases_over_gq_threshold=(9, 9), sum_dp=159, fraction_over_gq_threshold=(1.0, 1.0), mean_dp=159 / 9),
        hl.Struct(bases_over_gq_threshold=(9, 9), sum_dp=98, fraction_over_gq_threshold=(1.0, 1.0), mean_dp=98 / 9),
        hl.Struct(bases_over_gq_threshold=(9, 9), sum_dp=72, fraction_over_gq_threshold=(1.0, 1.0), mean_dp=8),
        hl.Struct(bases_over_gq_threshold=(9, 0), sum_dp=20, fraction_over_gq_threshold=(1.0, 0.0), mean_dp=2 / 9),
    ]

    for i in range(len(expected)):
        obs = observed[i]
        exp = expected[i]
        assert obs.bases_over_gq_threshold == exp.bases_over_gq_threshold, i
        assert obs.sum_dp == exp.sum_dp, i
        assert obs.fraction_over_gq_threshold == exp.fraction_over_gq_threshold, i
        pytest.approx(obs.mean_dp, exp.mean_dp)


@fails_local_backend
@fails_service_backend
def test_impute_sex_chromosome_ploidy():
    x_par_end = 2699521
    y_par_end = 2649521
    rg = hl.get_reference('GRCh37')
    ref_blocks = [
        hl.Struct(s='sample_xx', ref_allele='A', locus=hl.Locus('22', 1000000, rg), END=2000000, GQ=15, DP=5),
        hl.Struct(s='sample_xx', ref_allele='A', locus=hl.Locus('X', x_par_end-10, rg), END=x_par_end+9, GQ=18, DP=6),
        hl.Struct(s='sample_xx', ref_allele='A', locus=hl.Locus('X', x_par_end+10, rg), END=x_par_end+29, GQ=15, DP=5),
        hl.Struct(s='sample_xy', ref_allele='A', locus=hl.Locus('22', 1000000, rg), END=2000000, GQ=15, DP=5),
        hl.Struct(s='sample_xy', ref_allele='A', locus=hl.Locus('X', x_par_end-10, rg), END=x_par_end+9, GQ=9, DP=3),
        hl.Struct(s='sample_xy', ref_allele='A', locus=hl.Locus('X', x_par_end+10, rg), END=x_par_end+29, GQ=6, DP=2),
        hl.Struct(s='sample_xy', ref_allele='A', locus=hl.Locus('Y', y_par_end-10, rg), END=y_par_end+9, GQ=12, DP=4),
        hl.Struct(s='sample_xy', ref_allele='A', locus=hl.Locus('Y', y_par_end + 10, rg), END=y_par_end + 29, GQ=9, DP=3),
    ]

    ref_mt = hl.Table.parallelize(ref_blocks,
                                  schema=hl.dtype('struct{s:str,locus:locus<GRCh37>,ref_allele:str,END:int32,GQ:int32,DP:int32}')) \
        .to_matrix_table(row_key=['locus'], row_fields=['ref_allele'], col_key=['s'])
    var_mt = hl.Table.parallelize([],
                                  schema=hl.dtype('struct{locus:locus<GRCh37>,alleles:array<str>,s:str,LA:array<int32>,LGT:call,GQ:int32}'))\
    .to_matrix_table(row_key=['locus', 'alleles'], col_key=['s'])

    vds = hl.vds.VariantDataset(ref_mt, var_mt)

    calling_intervals = [
        hl.parse_locus_interval('22:1000010-1000020', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'X:{x_par_end}-{x_par_end+20}', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'Y:{y_par_end}-{y_par_end+20}', reference_genome='GRCh37'),
    ]

    r = hl.vds.impute_sex_chromosome_ploidy(vds, calling_intervals, normalization_contig='22')

    assert r.collect() == [
        hl.Struct(s='sample_xx',
                  autosomal_mean_dp=5.0,
                  x_mean_dp=5.5,
                  x_ploidy=2.2,
                  y_mean_dp=0.0,
                  y_ploidy=0.0),
        hl.Struct(s='sample_xy',
                  autosomal_mean_dp=5.0,
                  x_mean_dp=2.5,
                  x_ploidy=1.0,
                  y_mean_dp=3.5,
                  y_ploidy=1.4)
    ]
