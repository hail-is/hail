from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import *
import hail.utils as utils
from hail.utils.misc import test_file
from hail.linalg import BlockMatrix

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
    
    def test_write(self):
        file1 = utils.new_temp_file()
        file2 = utils.new_temp_file()
        
        ds = self.get_dataset()
        bm = BlockMatrix.from_matrix_table(functions.or_else(ds.GT.num_alt_alleles(), 0), block_size=50)
        gram = bm.dot(bm.T).cache()
        gram.write(file1, blocks_to_keep=[1, 3, 5])
        gram.write_band(file2, lower_bandwidth=25, upper_bandwidth=0, force_row_major=True)
