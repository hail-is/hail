import os

import hail as hl
from hail.utils import new_temp_file
from ..helpers import startTestHailContext, stopTestHailContext, resource, fails_local_backend, fails_service_backend

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


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


@fails_local_backend
@fails_service_backend
def test_combiner_works(self):
    from hail.vds.combiner import combine_varant_datasets, transform_gvcf
    _paths = ['gvcfs/HG00096.g.vcf.gz', 'gvcfs/HG00268.g.vcf.gz']
    paths = [resource(p) for p in _paths]
    parts = [
        hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 17821257, reference_genome='GRCh38')),
                    end=hl.Struct(locus=hl.Locus('chr20', 18708366, reference_genome='GRCh38')),
                    includes_end=True),
        hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 18708367, reference_genome='GRCh38')),
                    end=hl.Struct(locus=hl.Locus('chr20', 19776611, reference_genome='GRCh38')),
                    includes_end=True),
        hl.Interval(start=hl.Struct(locus=hl.Locus('chr20', 19776612, reference_genome='GRCh38')),
                    end=hl.Struct(locus=hl.Locus('chr20', 21144633, reference_genome='GRCh38')),
                    includes_end=True)
    ]
    vcfs = [transform_gvcf(mt.annotate_rows(info=mt.info.annotate(
        MQ_DP=hl.missing(hl.tint32),
        VarDP=hl.missing(hl.tint32),
        QUALapprox=hl.missing(hl.tint32))))
            for mt in hl.import_gvcfs(paths, parts, reference_genome='GRCh38',
                                      array_elements_required=False)]
    comb = combine_varant_datasets(vcfs)
    self.assertEqual(len(parts), comb.variant_data.n_partitions())
    vds_path = new_temp_file()
    comb.write(vds_path)
