import os
import subprocess as sp
import unittest

import pytest

import hail as hl
from hail import utils

from ...helpers import get_dataset, qobtest, test_timeout


def plinkify(ds, min=None, max=None):
    vcf = utils.new_temp_file(prefix="plink", extension="vcf")
    hl.export_vcf(ds, vcf)

    local_tmpdir = utils.new_local_temp_dir()
    plinkpath = f'{local_tmpdir}/plink-ibd'
    local_vcf = f'{local_tmpdir}/input.vcf'

    hl.hadoop_copy(vcf, local_vcf)

    threshold_string = "{} {}".format("--min {}".format(min) if min else "", "--max {}".format(max) if max else "")

    plink_command = "plink --double-id --allow-extra-chr --vcf {} --genome full --out {} {}".format(
        utils.uri_path(local_vcf), utils.uri_path(plinkpath), threshold_string
    )
    result_file = utils.uri_path(plinkpath + ".genome")

    sp.run(plink_command, check=True, capture_output=True, shell=True)

    ### format of .genome file is:
    # _, fid1, iid1, fid2, iid2, rt, ez, z0, z1, z2, pihat, phe,
    # dst, ppc, ratio, ibs0, ibs1, ibs2, homhom, hethet (+ separated)

    ### format of ibd is:
    # i (iid1), j (iid2), ibd: {Z0, Z1, Z2, PI_HAT}, ibs0, ibs1, ibs2
    results = {}
    with open(result_file) as f:
        f.readline()
        for line in f:
            row = line.strip().split()
            results[(row[1], row[3])] = (list(map(float, row[6:10])), list(map(int, row[14:17])))
    return results


@qobtest
@unittest.skipIf('HAIL_TEST_SKIP_PLINK' in os.environ, 'Skipping tests requiring plink')
@test_timeout(local=10 * 60, batch=10 * 60)
def test_ibd_default_arguments():
    ds = get_dataset()

    plink_results = plinkify(ds)
    hail_results = hl.identity_by_descent(ds).collect()

    for row in hail_results:
        key = (row.i, row.j)
        assert plink_results[key][0][0] == pytest.approx(row.ibd.Z0, abs=0.5e-4)
        assert plink_results[key][0][1] == pytest.approx(row.ibd.Z1, abs=0.5e-4)
        assert plink_results[key][0][2] == pytest.approx(row.ibd.Z2, abs=0.5e-4)
        assert plink_results[key][0][3] == pytest.approx(row.ibd.PI_HAT, abs=0.5e-4)
        assert plink_results[key][1][0] == row.ibs0
        assert plink_results[key][1][1] == row.ibs1
        assert plink_results[key][1][2] == row.ibs2


@unittest.skipIf('HAIL_TEST_SKIP_PLINK' in os.environ, 'Skipping tests requiring plink')
@test_timeout(local=10 * 60, batch=10 * 60)
def test_ibd_0_and_1():
    ds = get_dataset()

    plink_results = plinkify(ds, min=0.0, max=1.0)
    hail_results = hl.identity_by_descent(ds).collect()

    for row in hail_results:
        key = (row.i, row.j)
        assert plink_results[key][0][0] == pytest.approx(row.ibd.Z0, abs=0.5e-4)
        assert plink_results[key][0][1] == pytest.approx(row.ibd.Z1, abs=0.5e-4)
        assert plink_results[key][0][2] == pytest.approx(row.ibd.Z2, abs=0.5e-4)
        assert plink_results[key][0][3] == pytest.approx(row.ibd.PI_HAT, abs=0.5e-4)
        assert plink_results[key][1][0] == row.ibs0
        assert plink_results[key][1][1] == row.ibs1
        assert plink_results[key][1][2] == row.ibs2


@test_timeout(local=10 * 60, batch=10 * 60)
def test_ibd_does_not_error_with_dummy_maf_float64():
    dataset = get_dataset()
    dataset = dataset.annotate_rows(dummy_maf=0.01)
    hl.identity_by_descent(dataset, dataset['dummy_maf'], min=0.0, max=1.0)


@test_timeout(local=10 * 60, batch=10 * 60)
def test_ibd_does_not_error_with_dummy_maf_float32():
    dataset = get_dataset()
    dataset = dataset.annotate_rows(dummy_maf=0.01)
    hl.identity_by_descent(dataset, hl.float32(dataset['dummy_maf']), min=0.0, max=1.0)
