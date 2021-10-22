import os

import hail as hl

from hail.utils.java import Env
from hail.utils.misc import new_temp_file
from hail.vds.combiner import combine_variant_datasets, new_combiner, load_combiner, transform_gvcf
from hail.vds.combiner.combine import defined_entry_fields
from ..helpers import startTestHailContext, stopTestHailContext, resource, fails_local_backend, fails_service_backend

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


@fails_local_backend
@fails_service_backend
def test_combiner_works():
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
    vcfs = hl.import_gvcfs(paths, parts, reference_genome='GRCh38', array_elements_required=False)
    entry_to_keep = defined_entry_fields(vcfs[0].filter_rows(hl.is_defined(vcfs[0].info.END)), 100_000) - {'GT', 'PGT', 'PL'}
    vcfs = [transform_gvcf(mt.annotate_rows(info=mt.info.annotate(
        MQ_DP=hl.missing(hl.tint32),
        VarDP=hl.missing(hl.tint32),
        QUALapprox=hl.missing(hl.tint32))),
                           reference_entry_fields_to_keep=entry_to_keep)
            for mt in vcfs]
    comb = combine_variant_datasets(vcfs)
    assert len(parts) == comb.variant_data.n_partitions()
    comb.variant_data._force_count_rows()
    comb.reference_data._force_count_rows()


@fails_local_backend
@fails_service_backend
def test_vcf_vds_combiner_equivalence():
    import hail.experimental.vcf_combiner.vcf_combiner as vcf
    import hail.vds.combiner as vds
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
    vcfs = [mt.annotate_rows(info=mt.info.annotate(
        MQ_DP=hl.missing(hl.tint32),
        VarDP=hl.missing(hl.tint32),
        QUALapprox=hl.missing(hl.tint32)))
            for mt in hl.import_gvcfs(paths, parts, reference_genome='GRCh38',
                                      array_elements_required=False)]
    entry_to_keep = defined_entry_fields(vcfs[0].filter_rows(hl.is_defined(vcfs[0].info.END)), 100_000) - {'GT', 'PGT', 'PL'}
    vds = vds.combine_variant_datasets([vds.transform_gvcf(mt, reference_entry_fields_to_keep=entry_to_keep) for mt in vcfs])
    smt = vcf.combine_gvcfs([vcf.transform_gvcf(mt) for mt in vcfs])
    smt_from_vds = hl.vds.to_merged_sparse_mt(vds).drop('RGQ')
    smt = smt.select_entries(*smt_from_vds.entry)  # harmonize fields and order
    smt = smt.key_rows_by('locus', 'alleles')
    assert smt._same(smt_from_vds)


def test_combiner_plan_round_trip_serialization():
    sample_names = all_samples[:5]
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', f'{s}.hg38.g.vcf.gz') for s in sample_names]
    plan_path = new_temp_file(extension='json')
    out_file = new_temp_file(extension='vds')
    plan = new_combiner(gvcf_paths=paths,
                        output_path=out_file,
                        temp_path=Env.hc()._tmpdir,
                        save_path=plan_path,
                        reference_genome='GRCh38',
                        use_exome_default_intervals=True,
                        branch_factor=2,
                        batch_size=2)
    plan.save()
    plan_loaded = load_combiner(plan_path)
    assert plan == plan_loaded

@fails_local_backend
@fails_service_backend
def test_combiner_run():

    tmpdir = new_temp_file()
    samples = all_samples[:5]

    input_paths = [resource(os.path.join('gvcfs', '1kg_chr22', f'{s}.hg38.g.vcf.gz')) for s in samples]
    final_paths_individual = [os.path.join(tmpdir, f'sample_{s}') for s in samples]
    final_path_1 = os.path.join(tmpdir, 'final1.vds')
    final_path_2 = os.path.join(tmpdir, 'final2.vds')

    parts = hl.eval([hl.parse_locus_interval('chr22:start-end', reference_genome='GRCh38')])

    for input_gvcf, path in zip(input_paths[:2], final_paths_individual[:2]):
        combiner = hl.vds.new_combiner(output_path=path, intervals=parts,
                                       temp_path=tmpdir,
                                       gvcf_paths=[input_gvcf],
                                       reference_genome='GRCh38')
        combiner.run()

    combiner = hl.vds.new_combiner(output_path=final_path_1, intervals=parts, temp_path=tmpdir,
                                   gvcf_paths=input_paths[2:], vds_paths=final_paths_individual[:2],
                                   reference_genome='GRCh38',
                                   branch_factor=2, batch_size=2)
    combiner.run()

    combiner2 = hl.vds.new_combiner(output_path=final_path_2, intervals=parts, temp_path=tmpdir,
                                    gvcf_paths=input_paths,
                                    reference_genome='GRCh38',
                                    branch_factor=2, batch_size=2)
    combiner2.run()

    assert hl.vds.read_vds(final_path_1)._same(hl.vds.read_vds(final_path_2))


@fails_local_backend
@fails_service_backend
def test_combiner_manual_filtration():
    sample_names = all_samples[:2]
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', f'{s}.hg38.g.vcf.gz') for s in sample_names]
    out_file = new_temp_file(extension='vds')
    plan = new_combiner(gvcf_paths=paths,
                        output_path=out_file,
                        temp_path=Env.hc()._tmpdir,
                        reference_genome='GRCh38',
                        use_exome_default_intervals=True,
                        gvcf_reference_entry_fields_to_keep=['GQ'],
                        gvcf_info_to_keep=['ExcessHet'],
                        force=True)

    assert plan.gvcf_info_to_keep == {'ExcessHet'}

    plan.run()
    vds = hl.vds.read_vds(out_file)
    assert list(vds.variant_data.gvcf_info) == ['ExcessHet']
    assert list(vds.reference_data.entry) == ['END', 'GQ']
