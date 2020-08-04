import unittest
from hailtop.batch.contrib import regenie as br
import os


class LocalTests(unittest.TestCase):
    def read(self, file):
        with open(file, 'r') as f:
            result = f.read().rstrip()
        return result

    def assert_same_file(self, file1, file2):
        assert self.read(file1).rstrip() == self.read(file2).rstrip()

    def test_regenie(self):
        os.chdir("hailtop/batch/contrib/regenie")
        out_prefix = "batch"
        args = br.parse_input_args(["--demo", "--out", out_prefix])
        br.regenie(args)

        out_log = f"{out_prefix}.log"
        out1 = f"{out_prefix}.test_bin_out_firth_Y1.regenie"
        out2 = f"{out_prefix}.test_bin_out_firth_Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert(len(self.read(out_log)) > 0)
        assert(self.read(out1) == self.read(expected))
        assert(len(self.read(out2)) > 0)

        os.remove(out_log)
        os.remove(out1)
        os.remove(out2)
