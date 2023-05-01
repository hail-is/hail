import os
import pytest

import hail as hl
from hail.utils import new_temp_file
from hail.vds.combiner.combine import defined_entry_fields
from ..helpers import resource, fails_local_backend, fails_service_backend


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

    svcr_readback = hl.vds.to_merged_sparse_mt(vds, ref_allele_function=lambda ht: svcr.key_rows_by('locus').index_rows(ht.locus).alleles[0])

    assert svcr._same(svcr_readback, reorder_fields=True)


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

    assert res.aggregate_cols(hl.all(
        *(hl.agg.all(res.sample_qc[field] == res.sample_qc_new[field]) for field in fields_to_test)
    ))


def test_sampleqc_gq_dp():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    sqc = hl.vds.sample_qc(vds)

    assert hl.eval(sqc.index_globals()) == hl.Struct(gq_bins=(0, 20, 60), dp_bins=(0, 1, 10, 20, 30))

    hg00320 = sqc.filter(sqc.s == 'HG00320').select('bases_over_gq_threshold', 'bases_over_dp_threshold').collect()[0]
    assert hg00320 == hl.Struct(s='HG00320',
                                bases_over_gq_threshold=(334822, 515, 82),
                                bases_over_dp_threshold=(334822, 10484, 388, 111, 52))


def test_sampleqc_singleton_r_ti_tv():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    sqc = hl.vds.sample_qc(vds)

    hg00313 = sqc.filter(sqc.s == 'HG00313').select('r_ti_tv_singleton', 'n_singleton_ti', 'n_singleton_tv').collect()[0]
    assert hg00313 == hl.Struct(s='HG00313',
                                r_ti_tv_singleton=4.0,
                                n_singleton_ti=4,
                                n_singleton_tv=1)



def test_filter_samples_and_merge():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    vds.reference_data = vds.reference_data.drop('ref_allele')

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


def test_filter_samples_array():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))

    samples = ['HG00187', 'HG00190']
    other = ['HG00308', 'HG00313', 'HG00320']

    filt_in = hl.vds.filter_samples(vds, samples, keep=True)
    filt_out = hl.vds.filter_samples(vds, samples, keep=False)

    assert filt_in.reference_data.s.collect() == samples
    assert filt_in.variant_data.s.collect() == samples

    assert filt_out.reference_data.s.collect() == other
    assert filt_out.variant_data.s.collect() == other


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


def test_interval_coverage():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))

    interval1 = 'chr22:10678825-10678835'
    interval2 = 'chr22:10678970-10678979'

    intervals = hl.Table.parallelize(
        list(hl.struct(interval=hl.parse_locus_interval(x, reference_genome='GRCh38')) for x in [interval1, interval2]),
        key='interval')

    checkpoint_path = new_temp_file()
    r = hl.vds.interval_coverage(vds, intervals, gq_thresholds=(1, 21), dp_thresholds=(0, 1, 6)).checkpoint(
        checkpoint_path)
    assert r.aggregate_rows(hl.agg.collect((hl.format('%s:%d-%d', r.interval.start.contig, r.interval.start.position,
                                                      r.interval.end.position), r.interval_size))) == [(interval1, 10),
                                                                                                       (interval2, 9)]

    observed = r.aggregate_entries(hl.agg.collect(r.entry))
    expected = [
        hl.Struct(bases_over_gq_threshold=(10, 0), bases_over_dp_threshold=(10, 10, 5), sum_dp=55,
                  fraction_over_gq_threshold=(1.0, 0.0), fraction_over_dp_threshold=(1.0, 1.0, 0.5), mean_dp=5.5),
        hl.Struct(bases_over_gq_threshold=(10, 0), bases_over_dp_threshold=(10, 10, 0), sum_dp=45,
                  fraction_over_gq_threshold=(1.0, 0.0), fraction_over_dp_threshold=(1.0, 1.0, 0), mean_dp=4.5),
        hl.Struct(bases_over_gq_threshold=(0, 0), bases_over_dp_threshold=(10, 0, 0), sum_dp=0,
                  fraction_over_gq_threshold=(0.0, 0.0), fraction_over_dp_threshold=(1.0, 0, 0), mean_dp=0),
        hl.Struct(bases_over_gq_threshold=(10, 0), bases_over_dp_threshold=(10, 10, 0), sum_dp=30,
                  fraction_over_gq_threshold=(1.0, 0.0), fraction_over_dp_threshold=(1.0, 1.0, 0.0), mean_dp=3.0),
        hl.Struct(bases_over_gq_threshold=(9, 0), bases_over_dp_threshold=(10, 10, 0), sum_dp=10,
                  fraction_over_gq_threshold=(0.9, 0.0), fraction_over_dp_threshold=(1.0, 1.0, 0.0), mean_dp=1.0),

        hl.Struct(bases_over_gq_threshold=(9, 9), bases_over_dp_threshold=(9, 9, 9), sum_dp=153,
                  fraction_over_gq_threshold=(1.0, 1.0), fraction_over_dp_threshold=(1.0, 1.0, 1.0), mean_dp=17.0),
        hl.Struct(bases_over_gq_threshold=(9, 9), bases_over_dp_threshold=(9, 9, 9), sum_dp=159,
                  fraction_over_gq_threshold=(1.0, 1.0), fraction_over_dp_threshold=(1.0, 1.0, 1.0), mean_dp=159 / 9),
        hl.Struct(bases_over_gq_threshold=(9, 9), bases_over_dp_threshold=(9, 9, 9), sum_dp=98,
                  fraction_over_gq_threshold=(1.0, 1.0), fraction_over_dp_threshold=(1.0, 1.0, 1.0), mean_dp=98 / 9),
        hl.Struct(bases_over_gq_threshold=(9, 9), bases_over_dp_threshold=(9, 9, 9), sum_dp=72,
                  fraction_over_gq_threshold=(1.0, 1.0), fraction_over_dp_threshold=(1.0, 1.0, 1.0), mean_dp=8),
        hl.Struct(bases_over_gq_threshold=(9, 0), bases_over_dp_threshold=(9, 9, 0), sum_dp=20,
                  fraction_over_gq_threshold=(1.0, 0.0), fraction_over_dp_threshold=(1.0, 1.0, 0.0), mean_dp=2 / 9),
    ]

    for i in range(len(expected)):
        obs = observed[i]
        exp = expected[i]
        assert obs.bases_over_gq_threshold == exp.bases_over_gq_threshold, i
        assert obs.sum_dp == exp.sum_dp, i
        assert obs.bases_over_dp_threshold == exp.bases_over_dp_threshold, i
        assert obs.fraction_over_gq_threshold == exp.fraction_over_gq_threshold, i
        assert obs.fraction_over_dp_threshold == exp.fraction_over_dp_threshold, i
        pytest.approx(obs.mean_dp, exp.mean_dp)


def test_impute_sex_chr_ploidy_from_interval_coverage():
    norm_interval_1 = hl.parse_locus_interval('20:10-30', reference_genome='GRCh37')
    norm_interval_2 = hl.parse_locus_interval('20:40-45', reference_genome='GRCh37')
    x_interval_1 = hl.parse_locus_interval('X:10-20', reference_genome='GRCh37')
    x_interval_2 = hl.parse_locus_interval('X:25-35', reference_genome='GRCh37')
    y_interval_1 = hl.parse_locus_interval('Y:10-20', reference_genome='GRCh37')
    y_interval_2 = hl.parse_locus_interval('Y:25-30', reference_genome='GRCh37')

    mt = hl.Table.parallelize([hl.Struct(s='sample_xx', interval=norm_interval_1, sum_dp=195),
                               hl.Struct(s='sample_xx', interval=norm_interval_2, sum_dp=55),
                               hl.Struct(s='sample_xx', interval=x_interval_1, sum_dp=95),
                               hl.Struct(s='sample_xx', interval=x_interval_2, sum_dp=85),
                               hl.Struct(s='sample_xy', interval=norm_interval_1, sum_dp=190),
                               hl.Struct(s='sample_xy', interval=norm_interval_2, sum_dp=85),
                               hl.Struct(s='sample_xy', interval=x_interval_1, sum_dp=61),
                               hl.Struct(s='sample_xy', interval=x_interval_2, sum_dp=49),
                               hl.Struct(s='sample_xy', interval=y_interval_1, sum_dp=54),
                               hl.Struct(s='sample_xy', interval=y_interval_2, sum_dp=45)],
                              schema=hl.dtype(
                                  'struct{s:str,interval:interval<locus<GRCh37>>,sum_dp:int32}')).to_matrix_table(
        row_key=['interval'], col_key=['s'])

    mt = mt.annotate_rows(interval_size=mt.interval.end.position - mt.interval.start.position)
    r = hl.vds.impute_sex_chr_ploidy_from_interval_coverage(mt, normalization_contig='20')

    assert r.collect() == [
        hl.Struct(s='sample_xx',
                  autosomal_mean_dp=10.0,
                  x_mean_dp=9.0,
                  x_ploidy=1.8,
                  y_mean_dp=0.0,
                  y_ploidy=0.0),
        hl.Struct(s='sample_xy',
                  autosomal_mean_dp=11.0,
                  x_mean_dp=5.5,
                  x_ploidy=1.0,
                  y_mean_dp=6.6,
                  y_ploidy=1.2)
    ]


def test_impute_sex_chromosome_ploidy():
    x_par_end = 2699521
    y_par_end = 2649521
    rg = hl.get_reference('GRCh37')
    ref_blocks = [
        hl.Struct(s='sample_xx', locus=hl.Locus('22', 1000000, rg), END=2000000, GQ=15, DP=5),
        hl.Struct(s='sample_xx', locus=hl.Locus('X', x_par_end-10, rg), END=x_par_end+9, GQ=18, DP=6),
        hl.Struct(s='sample_xx', locus=hl.Locus('X', x_par_end+10, rg), END=x_par_end+29, GQ=15, DP=5),
        hl.Struct(s='sample_xy', locus=hl.Locus('22', 1000000, rg), END=2000000, GQ=15, DP=5),
        hl.Struct(s='sample_xy', locus=hl.Locus('X', x_par_end-10, rg), END=x_par_end+9, GQ=9, DP=3),
        hl.Struct(s='sample_xy', locus=hl.Locus('X', x_par_end+10, rg), END=x_par_end+29, GQ=6, DP=2),
        hl.Struct(s='sample_xy', locus=hl.Locus('Y', y_par_end-10, rg), END=y_par_end+9, GQ=12, DP=4),
        hl.Struct(s='sample_xy', locus=hl.Locus('Y', y_par_end+10, rg), END=y_par_end+29, GQ=9, DP=3),
    ]
    var = [
        hl.Struct(locus=hl.Locus('22', 2000021, rg), alleles=hl.array(["A", "C"]), s="sample_xx", LA=hl.array([0, 1]),
                  LGT=hl.call(0, 1, phased=False), GQ=15, DP=5),
        hl.Struct(locus=hl.Locus('X', x_par_end-11, rg), alleles=hl.array(["A", "C"]), s="sample_xx", LA=hl.array([0, 1]),
                  LGT=hl.call(0, 1, phased=False), GQ=18, DP=6),
        hl.Struct(locus=hl.Locus('X', x_par_end+30, rg), alleles=hl.array(["A", "C"]), s="sample_xx",
                  LA=hl.array([0, 1]),
                  LGT=hl.call(0, 1, phased=False), GQ=18, DP=6),
        hl.Struct(locus=hl.Locus('X', x_par_end + 33, rg), alleles=hl.array(["A", "C", "G"]), s="sample_xx",
                  LA=hl.array([0, 1, 2]),
                  LGT=hl.call(0, 2, phased=False), GQ=15, DP=5),
        hl.Struct(locus=hl.Locus('22', 2000021, rg), alleles=hl.array(["A", "C"]), s="sample_xy", LA=hl.array([0, 1]),
                  LGT=hl.call(0, 1, phased=False), GQ=15, DP=5),
        hl.Struct(locus=hl.Locus('X', x_par_end - 11, rg), alleles=hl.array(["A", "C"]), s="sample_xy",
                  LA=hl.array([0, 1]),
                  LGT=hl.call(1, 1, phased=False), GQ=5, DP=2),
        hl.Struct(locus=hl.Locus('X', x_par_end + 30, rg), alleles=hl.array(["A", "C"]), s="sample_xy",
                  LA=hl.array([0, 1]),
                  LGT=hl.call(1, 1, phased=False), GQ=7, DP=4),
        hl.Struct(locus=hl.Locus('X', x_par_end + 33, rg), alleles=hl.array(["A", "C", "G"]), s="sample_xy",
                  LA=hl.array([0, 1, 2]),
                  LGT=hl.call(2, 2, phased=False), GQ=5, DP=3),
        hl.Struct(locus=hl.Locus('Y', y_par_end-11, rg), alleles=hl.array(["A", "C"]), s="sample_xy", LA=hl.array([0, 1]),
                  LGT=hl.call(1, 1, phased=False), GQ=9, DP=2),
        hl.Struct(locus=hl.Locus('Y', y_par_end+30, rg), alleles=hl.array(["A", "C"]), s="sample_xy", LA=hl.array([0, 1]),
                  LGT=hl.call(1, 1, phased=False), GQ=12, DP=4),
        hl.Struct(locus=hl.Locus('Y', y_par_end+33, rg), alleles=hl.array(["A", "C"]), s="sample_xy",
                  LA=hl.array([0, 1]),
                  LGT=hl.call(1, 1, phased=False), GQ=6, DP=2),
    ]

    ref_mt = hl.Table.parallelize(ref_blocks,
                                  schema=hl.dtype('struct{s:str,locus:locus<GRCh37>,END:int32,GQ:int32,DP:int32}')) \
        .to_matrix_table(row_key=['locus'], row_fields=[], col_key=['s'])
    var_mt = hl.Table.parallelize([],
                                  schema=hl.dtype('struct{locus:locus<GRCh37>,alleles:array<str>,s:str,LA:array<int32>,LGT:call,GQ:int32,DP:int32}'))\
        .to_matrix_table(row_key=['locus', 'alleles'], col_key=['s'])

    vds = hl.vds.VariantDataset(ref_mt, var_mt)

    calling_intervals = [
        hl.parse_locus_interval('22:1000010-1000020', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'X:{x_par_end}-{x_par_end + 20}', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'Y:{y_par_end}-{y_par_end + 20}', reference_genome='GRCh37'),
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

    var_mt = hl.Table.parallelize(var,
                                  schema=hl.dtype('struct{locus:locus<GRCh37>,alleles:array<str>,s:str,LA:array<int32>,LGT:call,GQ:int32,DP:int32}'))\
        .to_matrix_table(row_key=['locus', 'alleles'], col_key=['s'])

    vds = hl.vds.VariantDataset(ref_mt, var_mt)

    calling_intervals = [
        hl.parse_locus_interval('22:1000010-1000020', reference_genome='GRCh37'),
        hl.parse_locus_interval('22:2000020-2000030', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'X:{x_par_end}-{x_par_end + 20}', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'X:{x_par_end + 32}-{x_par_end + 40}', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'Y:{y_par_end}-{y_par_end + 20}', reference_genome='GRCh37'),
        hl.parse_locus_interval(f'Y:{y_par_end + 32}-{y_par_end + 40}', reference_genome='GRCh37'),
    ]
    r = hl.vds.impute_sex_chromosome_ploidy(vds, calling_intervals, normalization_contig='22', use_variant_dataset=True)

    assert r.collect() == [
        hl.Struct(s='sample_xx',
                  autosomal_mean_dp=5.0,
                  x_mean_dp=5.0,
                  x_ploidy=2.0,
                  y_mean_dp=0.0,
                  y_ploidy=0.0),
        hl.Struct(s='sample_xy',
                  autosomal_mean_dp=5.0,
                  x_mean_dp=3.0,
                  x_ploidy=1.2,
                  y_mean_dp=2.0,
                  y_ploidy=0.8)
    ]


def test_filter_intervals_segment():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_2samples_starts.vds'))

    intervals = [hl.parse_locus_interval('chr22:10514784-10517000', reference_genome='GRCh38')]
    filt = hl.vds.filter_intervals(vds, intervals, split_reference_blocks=True)

    assert hl.vds.to_dense_mt(filt)._same(
        hl.filter_intervals(hl.vds.to_dense_mt(vds), intervals))

    ref = filt.reference_data
    var = filt.variant_data
    assert ref.aggregate_rows(hl.agg.all(intervals[0].contains(ref.locus)))
    assert var.aggregate_rows(hl.agg.all(intervals[0].contains(var.locus)))


def test_filter_intervals_segment_table():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_2samples_starts.vds'))

    intervals = [hl.parse_locus_interval('chr22:10514784-10517000', reference_genome='GRCh38')]
    intervals_table = hl.Table.parallelize(
        hl.array(intervals).map(lambda x: hl.struct(interval=x)),
        key='interval')
    filt = hl.vds.filter_intervals(vds, intervals_table, split_reference_blocks=True)

    assert hl.vds.to_dense_mt(filt)._same(
        hl.filter_intervals(hl.vds.to_dense_mt(vds), intervals))

    ref = filt.reference_data
    var = filt.variant_data
    assert ref.aggregate_rows(hl.agg.all(intervals[0].contains(ref.locus)))
    assert var.aggregate_rows(hl.agg.all(intervals[0].contains(var.locus)))


def test_filter_intervals_default():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_2samples_starts.vds'))

    intervals = [hl.parse_locus_interval('chr22:10514784-10517000', reference_genome='GRCh38')]
    filt = hl.vds.filter_intervals(vds, intervals)

    assert hl.vds.to_dense_mt(filt)._same(
        hl.filter_intervals(hl.vds.to_dense_mt(vds), intervals))

    var = filt.variant_data
    assert var.aggregate_rows(hl.agg.all(intervals[0].contains(var.locus)))


def test_filter_intervals_default_table():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_2samples_starts.vds'))

    intervals = [hl.parse_locus_interval('chr22:10514784-10517000', reference_genome='GRCh38')]
    intervals_table = hl.Table.parallelize(
        hl.array(intervals).map(lambda x: hl.struct(interval=x)),
        key='interval')
    filt = hl.vds.filter_intervals(vds, intervals_table)

    assert hl.vds.to_dense_mt(filt)._same(
        hl.filter_intervals(hl.vds.to_dense_mt(vds), intervals))

    var = filt.variant_data
    assert var.aggregate_rows(hl.agg.all(intervals[0].contains(var.locus)))


def test_filter_chromosomes():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_2samples_starts.vds'))


    autosomes = [f'chr{i}' for i in range(1, 23)]
    sex_chrs = ['chrX', 'chrY']

    all_chrs = autosomes + sex_chrs
    def assert_contigs(vds, expected):
        expected_set = set(expected)

        rd = vds.reference_data
        vd = vds.variant_data

        assert rd.aggregate_rows(hl.agg.collect_as_set(rd.locus.contig)) == expected_set
        assert vd.aggregate_rows(hl.agg.collect_as_set(vd.locus.contig)) == expected_set

    assert_contigs(vds, all_chrs)

    vds_keep_1_5 = hl.vds.filter_chromosomes(vds, keep=['chr1', 'chr5'])
    assert_contigs(vds_keep_1_5, ['chr1', 'chr5'])

    vds_remove_1_5 = hl.vds.filter_chromosomes(vds, remove=['chr1', 'chr5'])
    assert_contigs(vds_remove_1_5, [x for x in all_chrs if x not in ['chr1', 'chr5']])

    vds_auto = hl.vds.filter_chromosomes(vds, keep_autosomes=True)
    assert_contigs(vds_auto, autosomes)


def test_to_dense_mt():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_2samples_starts.vds'))
    vds = hl.vds.filter_chromosomes(vds, keep='chr22')

    dense = hl.vds.to_dense_mt(vds).select_entries('LGT', 'LA', 'GQ', 'DP')

    assert dense.rows().select()._same(
        vds.variant_data.rows().select()), "rows differ between variant data and dense mt"

    assert dense.filter_entries(hl.is_defined(dense.LA))._same(
        vds.variant_data.select_entries('LGT', 'LA', 'GQ', 'DP')), "cannot recover variant data"

    as_dict = dense.aggregate_entries(
        hl.dict(hl.zip(hl.agg.collect((hl.str(dense.locus), dense.s)), hl.agg.collect(dense.entry))))

    assert as_dict.get(('chr22:10514784', 'NA12891')) == None
    assert as_dict.get(('chr22:10514784', 'NA12878')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=23, DP=4)

    assert as_dict.get(('chr22:10516102', 'NA12891')) == hl.Struct(LGT=hl.Call([0, 0]), LA=None, GQ=12, DP=7)
    assert as_dict.get(('chr22:10516102', 'NA12878')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=26, DP=3)

    assert as_dict.get(('chr22:10516150', 'NA12891')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=64, DP=4)
    assert as_dict.get(('chr22:10516150', 'NA12878')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=99, DP=10)

    assert as_dict.get(('chr22:10519088', 'NA12891')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=99, DP=21)
    assert as_dict.get(('chr22:10519088', 'NA12878')) == None

    assert as_dict.get(('chr22:10557694', 'NA12891')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=28, DP=19)
    assert as_dict.get(('chr22:10557694', 'NA12878')) == hl.Struct(LGT=hl.Call([0, 0]), LA=None, GQ=13, DP=16)

    assert as_dict.get(('chr22:10562435', 'NA12891')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=99, DP=15)
    assert as_dict.get(('chr22:10562435', 'NA12878')) == hl.Struct(LGT=hl.Call([0, 0]), LA=None, GQ=21, DP=9)

    assert as_dict.get(('chr22:10562436', 'NA12891')) == hl.Struct(LGT=hl.Call([0, 1]), LA=[0, 1], GQ=99, DP=15)
    assert as_dict.get(('chr22:10562436', 'NA12878')) == hl.Struct(LGT=hl.Call([0, 0]), LA=None, GQ=21, DP=9)


def test_merge_reference_blocks():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    vds = hl.vds.filter_samples(vds, ['HG00187'])
    rd = vds.reference_data
    vds.reference_data = rd.annotate_entries(GQ=rd.GQ - rd.GQ % 10)
    vds.reference_data = vds.reference_data.annotate_entries(
        LEN=vds.reference_data.END - vds.reference_data.locus.position + 1, N_BLOCKS=1)
    sampqc_1 = hl.vds.sample_qc(vds, gq_bins=(0, 10, 20, 30), dp_bins=()).select('bases_over_gq_threshold')

    merged = hl.vds.merge_reference_blocks(vds,
                                           equivalence_function=lambda b1, b2: b1.GQ == b2.GQ,
                                           merge_functions={'LEN': 'sum',
                                                            'MIN_DP': 'min',
                                                            'N_BLOCKS': 'sum'}
                                           ).checkpoint(new_temp_file(extension='vds'))
    sampqc_2 = hl.vds.sample_qc(merged, gq_bins=(0, 10, 20, 30), dp_bins=()).select('bases_over_gq_threshold')

    assert sampqc_1._same(sampqc_2), "gq bins aren't the same"

    mrd = merged.reference_data
    n_blocks_original = vds.reference_data.entries().count()
    (n_blocks_merged, sum_n) = mrd.aggregate_entries((hl.agg.count(), hl.agg.sum(mrd.N_BLOCKS)))

    assert n_blocks_original == sum_n
    assert mrd.aggregate_entries(hl.agg.all(mrd.LEN == mrd.END - mrd.locus.position + 1))
    assert hl.vds.to_dense_mt(vds)._same(hl.vds.to_dense_mt(merged))


def test_truncate_reference_blocks():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    rd = vds.reference_data
    rd = rd.filter_entries(rd.GQ > 0)
    vds.reference_data = rd

    sampqc_1 = hl.vds.sample_qc(vds, gq_bins=(0, 10, 20, 30), dp_bins=()).select('bases_over_gq_threshold')

    max_bases = 20
    vds_trunc = hl.vds.truncate_reference_blocks(vds, max_ref_block_base_pairs=max_bases)
    sampqc_2 = hl.vds.sample_qc(vds_trunc, gq_bins=(0, 10, 20, 30), dp_bins=()).select('bases_over_gq_threshold')

    rd2 = vds_trunc.reference_data

    assert rd2.aggregate_entries(hl.agg.all(rd2.END - rd2.locus.position + 1 <= max_bases)), "long ref blocks remain"
    assert sampqc_1._same(sampqc_2), "gq bins aren't the same"
    assert hl.vds.to_dense_mt(vds)._same(hl.vds.to_dense_mt(vds_trunc))


def test_union_rows():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))

    vds1 = hl.vds.filter_intervals(vds,
                                   [hl.parse_locus_interval('chr22:start-10754094', reference_genome='GRCh38')],
                                   split_reference_blocks=True)
    vds2 = hl.vds.filter_intervals(vds,
                                   [hl.parse_locus_interval('chr22:10754094-end', reference_genome='GRCh38')],
                                   split_reference_blocks=True)


    vds_union = vds1.union_rows(vds2)
    assert hl.vds.to_dense_mt(vds)._same(hl.vds.to_dense_mt(vds_union))

    vds1_trunc = hl.vds.truncate_reference_blocks(vds1, max_ref_block_base_pairs=50)
    vds2_trunc = hl.vds.truncate_reference_blocks(vds1, max_ref_block_base_pairs=75)

    vds_trunc_union = vds1_trunc.union_rows(vds2_trunc)
    assert hl.eval(vds_trunc_union.reference_data.index_globals()[hl.vds.VariantDataset.ref_block_max_length_field]) == 75

    assert 'max_ref_block_length' not in vds1_trunc.union_rows(vds2).reference_data.globals


def test_combiner_max_len():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    all_samples = vds.reference_data.s.collect()
    samp1 = all_samples[:len(all_samples)//2]
    samp2 = all_samples[len(all_samples)//2:]

    vds1 = hl.vds.filter_samples(vds, samp1, remove_dead_alleles=True)
    vds2 = hl.vds.filter_samples(vds, samp2, remove_dead_alleles=True)

    vds1_trunc = hl.vds.truncate_reference_blocks(vds1, max_ref_block_base_pairs=50)
    vds2_trunc = hl.vds.truncate_reference_blocks(vds2, max_ref_block_base_pairs=75)

    from hail.vds.combiner.combine import combine_references

    combined1 = combine_references([vds1_trunc.reference_data, vds2_trunc.reference_data])
    assert hl.eval(combined1.index_globals()[hl.vds.VariantDataset.ref_block_max_length_field]) == 75

    combined2 = combine_references([vds1_trunc.reference_data, vds2.reference_data])
    assert hl.vds.VariantDataset.ref_block_max_length_field not in combined2.globals


def test_split_sparse_roundtrip():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    smt = hl.vds.to_merged_sparse_mt(vds)
    smt = hl.experimental.sparse_split_multi(smt)
    vds2 = hl.vds.VariantDataset.from_merged_representation(smt,
                                                            ref_block_fields=list(vds.reference_data.entry),
                                                            is_split=True)

    vds_split = hl.vds.split_multi(vds)
    assert vds2.variant_data.select_entries(*vds_split.variant_data.entry)._same(vds_split.variant_data)
    assert vds2.reference_data._same(vds_split.reference_data.drop('ref_allele'))


def test_ref_block_max_len_patch():
    vds = hl.vds.read_vds(os.path.join(resource('vds'), '1kg_chr22_5_samples.vds'))
    if 'ref_block_max_len' in vds.reference_data.globals:
        vds.reference_data = vds.reference_data.drop('ref_block_max_len')

    max_rb_len = vds.reference_data.aggregate_entries(
        hl.agg.max(vds.reference_data.END - vds.reference_data.locus.position + 1))
    with hl.TemporaryDirectory() as tmpdir:
        vds_path = os.path.join(tmpdir, 'to_patch.vds')
        vds.write(vds_path)

        hl.vds.store_ref_block_max_length(vds_path)

        vds2 = hl.vds.read_vds(vds_path)
        assert hl.eval(vds2.reference_data.index_globals()[hl.vds.VariantDataset.ref_block_max_length_field]) == max_rb_len
