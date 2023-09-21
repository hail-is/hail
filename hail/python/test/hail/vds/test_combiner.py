import os
import pytest

import hail as hl

from hail.utils.java import Env
from hail.utils.misc import new_temp_file
from hail.vds.combiner import combine_variant_datasets, new_combiner, load_combiner, transform_gvcf
from hail.vds.combiner.combine import defined_entry_fields
from ..helpers import resource, skip_when_service_backend, test_timeout, qobtest

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


@qobtest
def test_combiner_works():
    _paths = ['gvcfs/HG00096.g.vcf.gz', 'gvcfs/HG00268.g.vcf.gz']
    paths = [resource(p) for p in _paths]
    parts = [
        hl.Interval(start=hl.Locus('chr20', 17821257, reference_genome='GRCh38'),
                    end=hl.Locus('chr20', 18708366, reference_genome='GRCh38'),
                    includes_end=True),
        hl.Interval(start=hl.Locus('chr20', 18708367, reference_genome='GRCh38'),
                    end=hl.Locus('chr20', 19776611, reference_genome='GRCh38'),
                    includes_end=True),
        hl.Interval(start=hl.Locus('chr20', 19776612, reference_genome='GRCh38'),
                    end=hl.Locus('chr20', 21144633, reference_genome='GRCh38'),
                    includes_end=True)
    ]
    with hl.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, 'out.vds')
        hl.vds.new_combiner(temp_path=tmpdir, output_path=out, gvcf_paths=paths, intervals=parts, reference_genome='GRCh38').run()
        comb = hl.vds.read_vds(out)

        # see https://github.com/hail-is/hail/issues/13367 for why these assertions are here
        assert 'LPGT' in comb.variant_data.entry
        assert comb.variant_data.LPGT.dtype == hl.tcall

        assert len(parts) == comb.variant_data.n_partitions()
        comb.variant_data._force_count_rows()
        comb.reference_data._force_count_rows()


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

def test_reload_combiner_plan():
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
    plan_loaded = new_combiner(gvcf_paths=paths,
                               output_path=out_file,
                               temp_path=Env.hc()._tmpdir,
                               save_path=plan_path,
                               reference_genome='GRCh38',
                               use_exome_default_intervals=True,
                               branch_factor=2,
                               batch_size=2)
    assert plan == plan_loaded

def test_move_load_combiner_plan():
    fs = hl.current_backend().fs
    sample_names = all_samples[:5]
    paths = [os.path.join(resource('gvcfs'), '1kg_chr22', f'{s}.hg38.g.vcf.gz') for s in sample_names]
    plan_path = new_temp_file(extension='json')
    out_file = new_temp_file(extension='vds')
    new_plan_path = new_temp_file(extension='json')
    plan = new_combiner(gvcf_paths=paths,
                        output_path=out_file,
                        temp_path=Env.hc()._tmpdir,
                        save_path=plan_path,
                        reference_genome='GRCh38',
                        use_exome_default_intervals=True,
                        branch_factor=2,
                        batch_size=2)
    plan.save()
    fs.copy(plan_path, new_plan_path)
    plan_loaded = load_combiner(new_plan_path)
    assert plan != plan_loaded
    plan._save_path = new_plan_path
    assert plan == plan_loaded


@test_timeout(10 * 60)
@skip_when_service_backend(reason='Combiner makes extensive use of the Backend API which are serviced by starting a Hail Batch job to execute them. This test will be too slow until we change the combiner to use many fewer executes.')
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

    assert plan._gvcf_info_to_keep == {'ExcessHet'}

    plan.run()
    vds = hl.vds.read_vds(out_file)
    assert list(vds.variant_data.gvcf_info) == ['ExcessHet']
    assert list(vds.reference_data.entry) == ['END', 'GQ']


@test_timeout(10 * 60)
def test_ref_block_max_len_propagates_in_combiner():
    gvcfs = ['NA21123.hg38.g.vcf.gz', 'NA21099.hg38.g.vcf.gz', 'NA19747.hg38.g.vcf.gz']
    with hl.TemporaryDirectory() as tmpdir:
        vds_paths = []
        for i, gvcf in enumerate(gvcfs):
            p = os.path.join(tmpdir, f'{i}.vds')
            vds_paths.append(p)
            c = hl.vds.new_combiner(output_path=p, temp_path=tmpdir,
                                    gvcf_paths=[os.path.join(resource('gvcfs'), '1kg_chr22', gvcf)],
                                    reference_genome='GRCh38',
                                    import_interval_size=1000000000)
            c.run()

        for path in vds_paths:
            vds = hl.vds.read_vds(path)
            assert hl.vds.VariantDataset.ref_block_max_length_field in vds.reference_data.globals
        final_path = os.path.join(tmpdir, 'final.vds')
        hl.vds.new_combiner(output_path=final_path, temp_path=tmpdir,
                            vds_paths=vds_paths,
                            reference_genome='GRCh38').run()
        vds = hl.vds.read_vds(final_path)
        assert hl.vds.VariantDataset.ref_block_max_length_field in vds.reference_data.globals


def test_custom_call_fields():
    _paths = ['gvcfs/HG00096.g.vcf.gz', 'gvcfs/HG00268.g.vcf.gz']
    paths = [resource(p) for p in _paths]
    parts = [
        hl.Interval(start=hl.Locus('chr20', 17821257, reference_genome='GRCh38'),
                    end=hl.Locus('chr20', 21144633, reference_genome='GRCh38'),
                    includes_end=True),
    ]
    with hl.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, 'out.vds')
        hl.vds.new_combiner(temp_path=tmpdir, output_path=out, gvcf_paths=paths, intervals=parts, call_fields=[], reference_genome='GRCh38').run()
        comb = hl.vds.read_vds(out)

        assert 'LPGT' in comb.variant_data.entry
        assert comb.variant_data.LPGT.dtype == hl.tstr
