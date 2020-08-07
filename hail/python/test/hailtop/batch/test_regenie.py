import unittest
import hailtop.batch.genetics.regenie as br
import os
from shutil import which, rmtree


rdir = "hailtop/batch/genetics/regenie"


def read(file):
    with open(file, 'r') as f:
        return f.read()


def assert_same_file(file1, file2):
    assert read(file1) == read(file2)


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.outdir = "batch"
        self.step2_out_prefix = 'test_bin_out_firth'

    @unittest.skipIf(not which("docker"), "docker command is missing")
    def test_regenie(self):
        os.chdir(rdir)

        args = br.parse_input_args(["--demo", "--outdir", self.outdir])
        br.run(args)

        out_log = f"{self.outdir}/{self.step2_out_prefix}.log"
        out1 = f"{self.outdir}/{self.step2_out_prefix}.Y1.regenie"
        out2 = f"{self.outdir}/{self.step2_out_prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert len(read(out_log)) > 0
        assert_same_file(out1, expected)
        assert len(read(out2)) > 0

        rmtree(self.outdir)

        os.chdir(self.cwd)

    @unittest.skipIf(not which("docker"), "docker command is missing")
    def test_regenie_1pheno(self):
        os.chdir(rdir)
        args = br.parse_input_args(["--local", "--step1", "example/step1.txt", "--step2",
                                    "example/step2-phenoCol.txt", "--outdir", self.outdir])
        br.run(args)

        out1 = f"{self.outdir}/{self.step2_out_prefix}.Y1.regenie"
        out2 = f"{self.outdir}/{self.step2_out_prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert_same_file(out1, expected)
        assert not os.path.isfile(out2)

        rmtree(self.outdir)

        args = br.parse_input_args(["--local", "--step1", "example/step1.txt", "--step2",
                                    "example/step2-phenoColList.txt", "--outdir", self.outdir])
        br.run(args)

        assert len(read(out2)) > 0
        assert not os.path.isfile(out1)

        rmtree(self.outdir)

        os.chdir(self.cwd)

    @unittest.skipIf(not which("docker"), "docker command is missing")
    def test_regenie_nosplit(self):
        os.chdir(rdir)
        args = br.parse_input_args(["--local", "--step1", "example/step1.txt", "--step2",
                                    "example/step2-nosplit.txt", "--outdir", self.outdir])
        br.run(args)

        out1 = f"{self.outdir}/{self.step2_out_prefix}.regenie"

        assert len(read(out1)) > 0

        rmtree(self.outdir)
