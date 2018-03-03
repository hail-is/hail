import unittest

import hail as hl
import hail.utils as utils
from hail.utils.misc import test_file
from hail.linalg import BlockMatrix

def setUpModule():
    hl.init(master='local[2]', min_block_size=0)

def tearDownModule():
    hl.stop()

class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = hl.split_multi_hts(hl.import_vcf(test_file('sample.vcf')))
        return Tests._dataset
    
    def test_write(self):
        file = utils.new_temp_file()

        ds = self.get_dataset()
        bm = BlockMatrix.from_matrix_table(hl.or_else(ds.GT.n_alt_alleles(), 0), block_size=32)
        bm.write(file)
