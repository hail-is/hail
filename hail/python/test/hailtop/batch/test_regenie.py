import unittest
import hailtop.batch.genetics.regenie as br
import os
from shutil import which, rmtree
from hailtop.config import get_user_config
import google.cloud.storage
import uuid


def read(file):
    with open(file, 'r') as f:
        return f.read()


def assert_same_file(file1, file2):
    assert read(file1) == read(file2)


class LocalBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rdir = "hailtop/batch/genetics/regenie"

        cls.cwd = os.getcwd()
        os.chdir(rdir)
        os.system("git clone --depth 1 --branch v1.0.5.6 https://github.com/rgcgithub/regenie.git")

        cls.outdir = "out"
        cls.step2_out_prefix = f'{cls.outdir}/test_bin_out_firth'

    @classmethod
    def tearDownClass(cls):
        rmtree('regenie')
        os.chdir(cls.cwd)

    @unittest.skipIf(not which("docker"), "docker command is missing")
    def test_regenie(self):
        args = br.parse_input_args(["--demo"])
        br.run(**args)

        out_log = f"{self.step2_out_prefix}.log"
        out1 = f"{self.step2_out_prefix}.Y1.regenie"
        out2 = f"{self.step2_out_prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert len(read(out_log)) > 0
        assert_same_file(out1, expected)
        assert len(read(out2)) > 0

        rmtree(self.outdir)

    @unittest.skipIf(not which("docker"), "docker command is missing")
    def test_regenie_1pheno(self):
        args = br.parse_input_args(["--local", "--step1", "example/step1.txt", "--step2",
                                    "example/step2-phenoCol.txt"])
        print("args", args)
        br.run(**args)

        out1 = f"{self.step2_out_prefix}.Y1.regenie"
        out2 = f"{self.step2_out_prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert_same_file(out1, expected)
        assert not os.path.isfile(out2)

        rmtree(self.outdir)

        args = br.parse_input_args(["--local", "--step1", "example/step1.txt", "--step2",
                                    "example/step2-phenoColList.txt"])
        br.run(**args)

        assert len(read(out2)) > 0
        assert not os.path.isfile(out1)

        rmtree(self.outdir)

    @unittest.skipIf(not which("docker"), "docker command is missing")
    def test_regenie_nosplit(self):
        args = br.parse_input_args(["--local", "--step1", "example/step1.txt", "--step2",
                                    "example/step2-nosplit.txt"])
        br.run(**args)

        out1 = f"{self.step2_out_prefix}.regenie"

        assert len(read(out1)) > 0

        rmtree(self.outdir)


class ServiceBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.system("git clone --depth 1 --branch v1.0.5.6 https://github.com/rgcgithub/regenie.git")
        cls.step2_out_prefix = 'test_bin_out_firth'

        cls.bucket_name = get_user_config().get('batch', 'bucket')

        print("self.bucket_name", cls.bucket_name)

        input_folder = 'batch-tests/resources/regenie'
        cls.gcs_input_dir = f'gs://{cls.bucket_name}/{input_folder}'

        token = uuid.uuid4()
        cls.gcs_output_path = f'batch-tests/{token}/'
        cls.gcs_output_dir = f'gs://{cls.bucket_name}/{cls.gcs_output_path}'

        in_cluster_key_file = '/test-gsa-key/key.json'
        if os.path.exists(in_cluster_key_file):
            credentials = google.oauth2.service_account.Credentials.from_service_account_file(
                in_cluster_key_file)
        else:
            credentials = None
        gcs_client = google.cloud.storage.Client(project='hail-vdc', credentials=credentials)
        bucket = gcs_client.bucket(cls.bucket_name)
        for file_name in os.listdir('regenie/example/'):
            file_path = os.path.join('regenie/example/', file_name)
            blob = bucket.blob(f'{input_folder}/{file_name}')
            if not blob.exists():
                blob.upload_from_filename(file_path)

        step1 = f"""
        --step 1
        --bed {cls.gcs_input_dir}/example
        --exclude {cls.gcs_input_dir}/snplist_rm.txt
        --covarFile {cls.gcs_input_dir}/covariates.txt
        --phenoFile {cls.gcs_input_dir}/phenotype_bin.txt
        --remove {cls.gcs_input_dir}/fid_iid_to_remove.txt
        --bsize 100
        --bt
        --lowmem
        --lowmem-prefix tmp_rg
        --out fit_bin_out
        """

        step2 = f"""
        --step 2
        --bgen {cls.gcs_input_dir}/example.bgen
        --covarFile {cls.gcs_input_dir}/covariates.txt
        --phenoFile {cls.gcs_input_dir}/phenotype_bin.txt
        --remove {cls.gcs_input_dir}/fid_iid_to_remove.txt
        --bsize 200
        --bt
        --firth --approx
        --pThresh 0.01
        --pred fit_bin_out_pred.list
        --out test_bin_out_firth
        """

        step2split = f"{step2}\n--split"
        step2pheno = f"{step2}\n--phenoColList Y2"

        cls.step1 = "step1-svc.txt"
        cls.step2 = "step2-nosplit-svc.txt"
        cls.step2pheno = "step2-phenolist-svc.txt"
        cls.step2split = "step2-split-svc.txt"

        with open(cls.step1) as f:
            f.write(step1)

        with open(cls.step2) as f:
            f.write(step2)

        with open(cls.step2split) as f:
            f.write(step2split)

        with open(cls.step2split) as f:
            f.write(step2pheno)

    def test_regenie(self):
        args = br.parse_input_args([
            "--step1", self.step1, "--step2", self.step2,
            "--outdir", self.gcs_output_dir,
            "--wait"])
        br.run(args)

        out_log = f"{self.gcs_output_dir}/{self.step2_out_prefix}.log"
        out1 = f"{self.gcs_output_dir}/{self.step2_out_prefix}.Y1.regenie"
        out2 = f"{self.gcs_output_dir}/{self.step2_out_prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert len(read(out_log)) > 0
        assert_same_file(out1, expected)
        assert len(read(out2)) > 0

        rmtree(self.outdir)
