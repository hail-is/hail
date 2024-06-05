import logging
import os
import subprocess
from pathlib import Path

import pytest

import hail as hl


@pytest.fixture(scope='session')
def resource_dir(request, tmpdir_factory):
    run_config = request.config.run_config
    if run_config.data_dir is not None:
        resource_dir = Path(run_config.data_dir)
        resource_dir.mkdir(parents=True, exist_ok=True)
    else:
        resource_dir = tmpdir_factory.mktemp('hail_benchmark_resources')

    return resource_dir


gs_curl_root = 'https://storage.googleapis.com/hail-common/benchmark'


def __download(data_dir, filename):
    url = os.path.join(gs_curl_root, filename)
    logging.info(f'downloading: {filename}')
    # Note: the below does not work on batch due to docker/ssl problems
    # dest = os.path.join(data_dir, filename)
    # urlretrieve(url, dest)
    subprocess.check_call(['curl', url, '-Lfs', '--output', f'{data_dir / filename}'])
    logging.info(f'done: {filename}')


def localize(path: Path):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        __download(path.parent, path.name)

    return path


@pytest.fixture(scope='session')
def empty_gvcf(resource_dir):
    path = resource_dir / 'empty_gvcf'
    _ = localize(path / 'empty.g.vcf.bgz.tbi')
    return localize(path / 'empty.g.vcf.bgz')


@pytest.fixture(scope='session')
def onekg_chr22(resource_dir):
    tar_path = localize(resource_dir / '1kg_chr22.tar')
    path = resource_dir / '1kg_chr22'
    path.mkdir(exist_ok=True)
    subprocess.check_call(['tar', '-xf', tar_path, '-C', str(path), '--strip', '1'])
    subprocess.check_call(['rm', tar_path])
    return path


def def_chr22_gvcf_fixture(name: str):
    @pytest.fixture(scope='session', name=name)
    def fix(onekg_chr22):
        return onekg_chr22 / f'{name}.hg38.g.vcf.gz'

    fix.__name__ = name
    globals()[name] = fix


def_chr22_gvcf_fixture('HG00308')
def_chr22_gvcf_fixture('HG00592')
def_chr22_gvcf_fixture('HG02230')
def_chr22_gvcf_fixture('NA18534')
def_chr22_gvcf_fixture('NA20760')
def_chr22_gvcf_fixture('NA18530')
def_chr22_gvcf_fixture('HG03805')
def_chr22_gvcf_fixture('HG02223')
def_chr22_gvcf_fixture('HG00637')
def_chr22_gvcf_fixture('NA12249')
def_chr22_gvcf_fixture('HG02224')
def_chr22_gvcf_fixture('NA21099')
def_chr22_gvcf_fixture('NA11830')
def_chr22_gvcf_fixture('HG01378')
def_chr22_gvcf_fixture('HG00187')
def_chr22_gvcf_fixture('HG01356')
def_chr22_gvcf_fixture('HG02188')
def_chr22_gvcf_fixture('NA20769')
def_chr22_gvcf_fixture('HG00190')
def_chr22_gvcf_fixture('NA18618')
def_chr22_gvcf_fixture('NA18507')
def_chr22_gvcf_fixture('HG03363')
def_chr22_gvcf_fixture('NA21123')
def_chr22_gvcf_fixture('HG03088')
def_chr22_gvcf_fixture('NA21122')
def_chr22_gvcf_fixture('HG00373')
def_chr22_gvcf_fixture('HG01058')
def_chr22_gvcf_fixture('HG00524')
def_chr22_gvcf_fixture('NA18969')
def_chr22_gvcf_fixture('HG03833')
def_chr22_gvcf_fixture('HG04158')
def_chr22_gvcf_fixture('HG03578')
def_chr22_gvcf_fixture('HG00339')
def_chr22_gvcf_fixture('HG00313')
def_chr22_gvcf_fixture('NA20317')
def_chr22_gvcf_fixture('HG00553')
def_chr22_gvcf_fixture('HG01357')
def_chr22_gvcf_fixture('NA19747')
def_chr22_gvcf_fixture('NA18609')
def_chr22_gvcf_fixture('HG01377')
def_chr22_gvcf_fixture('NA19456')
def_chr22_gvcf_fixture('HG00590')
def_chr22_gvcf_fixture('HG01383')
def_chr22_gvcf_fixture('HG00320')
def_chr22_gvcf_fixture('HG04001')
def_chr22_gvcf_fixture('NA20796')
def_chr22_gvcf_fixture('HG00323')
def_chr22_gvcf_fixture('HG01384')
def_chr22_gvcf_fixture('NA18613')
def_chr22_gvcf_fixture('NA20802')


@pytest.fixture(scope='session')
def single_gvcf(NA20760):
    return NA20760


@pytest.fixture(scope='session')
def chr22_gvcfs(
    HG00308,
    HG00592,
    HG02230,
    NA18534,
    NA20760,
    NA18530,
    HG03805,
    HG02223,
    HG00637,
    NA12249,
    HG02224,
    NA21099,
    NA11830,
    HG01378,
    HG00187,
    HG01356,
    HG02188,
    NA20769,
    HG00190,
    NA18618,
    NA18507,
    HG03363,
    NA21123,
    HG03088,
    NA21122,
    HG00373,
    HG01058,
    HG00524,
    NA18969,
    HG03833,
    HG04158,
    HG03578,
    HG00339,
    HG00313,
    NA20317,
    HG00553,
    HG01357,
    NA19747,
    NA18609,
    HG01377,
    NA19456,
    HG00590,
    HG01383,
    HG00320,
    HG04001,
    NA20796,
    HG00323,
    HG01384,
    NA18613,
    NA20802,
):
    return [
        HG00308,
        HG00592,
        HG02230,
        NA18534,
        NA20760,
        NA18530,
        HG03805,
        HG02223,
        HG00637,
        NA12249,
        HG02224,
        NA21099,
        NA11830,
        HG01378,
        HG00187,
        HG01356,
        HG02188,
        NA20769,
        HG00190,
        NA18618,
        NA18507,
        HG03363,
        NA21123,
        HG03088,
        NA21122,
        HG00373,
        HG01058,
        HG00524,
        NA18969,
        HG03833,
        HG04158,
        HG03578,
        HG00339,
        HG00313,
        NA20317,
        HG00553,
        HG01357,
        NA19747,
        NA18609,
        HG01377,
        NA19456,
        HG00590,
        HG01383,
        HG00320,
        HG04001,
        NA20796,
        HG00323,
        HG01384,
        NA18613,
        NA20802,
    ]


@pytest.fixture(scope='session')
def profile25_vcf(resource_dir):
    return localize(resource_dir / 'profile25' / 'profile.vcf.bgz')


@pytest.fixture(scope='session')
def profile25_mt(resource_dir, profile25_vcf):
    path = resource_dir / 'profile.mt'
    mt = hl.import_vcf(str(profile25_vcf), min_partitions=16)
    mt.write(str(path), overwrite=True)
    return path


@pytest.fixture(scope='session')
def gnomad_dp_sim(resource_dir):
    path = resource_dir / 'gnomad_dp_simulation.mt'
    mt = hl.utils.range_matrix_table(n_rows=250_000, n_cols=1_000, n_partitions=32)
    mt = mt.annotate_entries(x=hl.int(hl.rand_unif(0, 4.5) ** 3))
    mt.write(str(path), overwrite=True)
    return path


@pytest.fixture(scope='session')
def sim_ukb_bgen(resource_dir):
    bgen = localize(resource_dir / 'sim_ukb' / 'sim_ukb.bgen')
    hl.index_bgen(str(bgen))
    return bgen


@pytest.fixture(scope='session')
def sim_ukb_sample(resource_dir):
    return localize(resource_dir / 'sim_ukb' / 'sim_ukb.sample')


@pytest.fixture(scope='session')
def random_doubles_tsv(resource_dir):
    return localize(resource_dir / 'random_doubles' / 'random_doubles_mt.tsv.bgz')


@pytest.fixture(scope='session')
def random_doubles_mt(random_doubles_tsv):
    path = random_doubles_tsv.parent / 'random_doubles_mt.mt'
    mt = hl.import_matrix_table(
        str(random_doubles_tsv),
        row_key="row_idx",
        row_fields={"row_idx": hl.tint32},
        entry_type=hl.tfloat64,
    )
    mt.write(str(path), overwrite=True)
    return path


@pytest.fixture(scope='session')
def many_ints_tsv(resource_dir):
    return localize(resource_dir / 'many_ints_table' / 'many_ints_table.tsv.bgz')


def ht_from_tsv(tsv, types):
    stem = tsv
    for _ in tsv.suffixes:
        stem = tsv.stem

    ht_path = tsv.parent / (stem + '.ht')
    hl.import_table(str(tsv), types=types).write(str(ht_path), overwrite=True)
    return ht_path


@pytest.fixture(scope='session')
def many_ints_ht(many_ints_tsv):
    return ht_from_tsv(
        many_ints_tsv,
        {'idx': 'int', **{f'i{i}': 'int' for i in range(5)}, **{f'array{i}': 'array<int>' for i in range(2)}},
    )


@pytest.fixture(scope='session')
def many_strings_tsv(resource_dir):
    return localize(resource_dir / 'many_strings_table' / 'many_strings_table.tsv.bgz')


@pytest.fixture(scope='session')
def many_strings_ht(many_strings_tsv):
    return ht_from_tsv(many_strings_tsv, types={})


@pytest.fixture(scope='session')
def balding_nichols_5k_5k(resource_dir):
    path = resource_dir / 'balding_nichols_5k_5k.mt'
    if not path.exists():
        hl.balding_nichols_model(n_populations=5, n_variants=5000, n_samples=5000, n_partitions=16).write(str(path))
    return path


@pytest.fixture(scope='session', params=[10, 100, 1_000])
def many_partitions_ht(resource_dir, request):
    n_partitions = request.param
    path = resource_dir / f'table_10M_par_{n_partitions}.ht'
    if not path.exists():
        hl.utils.range_table(10_000_000, n_partitions).write(str(path))

    return path
