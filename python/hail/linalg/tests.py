from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import *
from subprocess import call as syscall
import numpy as np
import hail.utils as utils
from hail.linalg import BlockMatrix # FIXME: needed?

def setUpModule():
    init(master='local[2]', min_block_size=0)


def tearDownModule():
    stop()


class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = methods.split_multi_hts(methods.import_vcf(test_file('sample.vcf')))
        return Tests._dataset
    
    def test_matrix(self):
        filename = utils.new_temp_file()

        ds = self.get_dataset()
        bm = BlockMatrix.from_matrix_table('ds.GT.numAltAlleles()', 10)
        gram = bm.dot(bm.T)
        gram.write_band(filename, lower_bandwidth=100, upper_bandwidth=0, force_row_major=True)
        