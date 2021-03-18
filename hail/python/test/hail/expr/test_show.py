from ..helpers import startTestHailContext, stopTestHailContext, fails_service_backend
import unittest

import hail as hl

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    @fails_service_backend()
    def test(self):
        mt = hl.balding_nichols_model(3, 10, 10)
        t = mt.rows()
        mt.GT.show()
        mt.locus.show()
        mt.af.show()
        mt.pop.show()
        mt.sample_idx.show()
        mt.bn.show()
        mt.bn.fst.show()
        mt.GT.n_alt_alleles().show()
        (mt.GT.n_alt_alleles() * mt.GT.n_alt_alleles()).show()
        (mt.af * mt.GT.n_alt_alleles()).show()
        t.af.show()
        (t.af * 3).show()

    def test_show_negative(self):
        hl.utils.range_table(5).show(-1)
