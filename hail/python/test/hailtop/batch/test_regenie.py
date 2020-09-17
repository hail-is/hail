import unittest
import hailtop.batch.genetics.regenie as br
import os
from shutil import which, rmtree
from hailtop.config import get_user_config
import google.cloud.storage
from google.cloud.storage.blob import Blob
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

        cls.bucket_name = get_user_config().get('batch', 'bucket')

        input_folder = 'batch-tests/resources/regenie-v1.0.5.6'
        cls.gcs_input_dir = f'gs://{cls.bucket_name}/{input_folder}'

        in_cluster_key_file = os.environ.get('HAIL_GSA_KEY_FILE')
        if in_cluster_key_file:
            credentials = google.oauth2.service_account.Credentials.from_service_account_file(
                in_cluster_key_file)
        else:
            credentials = None
        cls.gcs_client = google.cloud.storage.Client(project='hail-vdc', credentials=credentials)
        cls.bucket = cls.gcs_client.bucket(cls.bucket_name)
        for file_name in os.listdir('regenie/example/'):
            file_path = os.path.join('regenie/example/', file_name)
            blob = cls.bucket.blob(f'{input_folder}/{file_name}')
            if not blob.exists():
                blob.upload_from_filename(file_path)

        token = uuid.uuid4()
        cls.gcs_output_path = f'batch-tests/{token}'
        cls.gcs_output_dir = f'gs://{cls.bucket_name}/{cls.gcs_output_path}'
        cls.step2prefix = f"{cls.gcs_output_dir}/test_bin_out_firth"

        step1 = f"""
        --step 1
        --threads 0.125
        --memory 375Mi
        --storage 1Gi
        --bed {cls.gcs_input_dir}/example
        --exclude {cls.gcs_input_dir}/snplist_rm.txt
        --covarFile {cls.gcs_input_dir}/covariates.txt
        --phenoFile {cls.gcs_input_dir}/phenotype_bin.txt
        --remove {cls.gcs_input_dir}/fid_iid_to_remove.txt
        --bsize 100
        --bt
        --out fit_bin_out
        """
        step1lowmen = f"{step1}\n--lowmem\n--lowmem-prefix tmp_rg"

        cls.step2 = f"""
        --step 2
        --threads 0.125
        --memory 375Mi
        --storage 1Gi
        --bgen {cls.gcs_input_dir}/example.bgen
        --covarFile {cls.gcs_input_dir}/covariates.txt
        --phenoFile {cls.gcs_input_dir}/phenotype_bin.txt
        --remove {cls.gcs_input_dir}/fid_iid_to_remove.txt
        --bsize 200
        --bt
        --firth --approx
        --pThresh 0.01
        --pred fit_bin_out_pred.list
        """
        cls.step2_split = f"{cls.step2}\n--split"

        cls.step1 = "step1-svc.txt"
        cls.step1low = "step1low-svc.txt"

        with open(cls.step1, 'w') as f:
            f.write(step1)

        with open(cls.step1low, 'w') as f:
            f.write(step1lowmen)

    def tearDown(self):
        blobs = self.bucket.list_blobs(prefix=self.gcs_output_dir)
        for blob in blobs:
            blob.delete()
        os.remove('step2.txt')

    @classmethod
    def read(cls, path):
        blob = Blob.from_string(path, cls.gcs_client)
        return blob.download_as_string().decode("utf-8")

    @classmethod
    def exists(cls, path):
        blob = Blob.from_string(path, cls.gcs_client)
        return blob.exists()

    def test_regenie_nosplit(self):
        step2prefix = f'{self.gcs_output_dir}/nosplit'
        with open('step2.txt', 'w') as f:
            f.write(f"{self.step2}\n--out {step2prefix}")

        args = br.parse_input_args(["--step1", self.step1, "--step2", 'step2.txt', "--wait"])
        res = br.run(**args)
        assert res.status()['state'] == "success"

        outpath = f"{step2prefix}.regenie"
        assert len(self.read(outpath)) > 0

    def test_regenie_nosplit_lowmem(self):
        step2prefix = f'{self.gcs_output_dir}/nosplit-lowmem'
        with open('step2.txt', 'w') as f:
            f.write(f"{self.step2}\n--out {step2prefix}")

        args = br.parse_input_args(["--step1", self.step1low, "--step2", 'step2.txt', "--wait"])
        res = br.run(**args)
        assert res.status()['state'] == "success"

        outpath = f"{step2prefix}.regenie"
        assert len(self.read(outpath)) > 0

    def test_regenie(self):
        step2prefix = f'{self.gcs_output_dir}/split'
        with open('step2.txt', 'w') as f:
            f.write(f"{self.step2_split}\n--out {step2prefix}")

        args = br.parse_input_args(["--step1", self.step1low, "--step2", 'step2.txt', "--wait"])
        res = br.run(**args)
        assert res.status()['state'] == "success"

        out_log = f"{step2prefix}.log"
        out1 = f"{step2prefix}.Y1.regenie"
        out2 = f"{step2prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert len(self.read(out_log)) > 0
        assert self.read(out1) == read(expected)
        assert len(self.read(out2)) > 0

    def test_regenie_1pheno(self):
        step2prefix = f'{self.gcs_output_dir}/phenoCol'
        with open('step2.txt', 'w') as f:
            f.write(f"{self.step2_split}\n--phenoCol Y1\n--out {step2prefix}")

        args = br.parse_input_args(["--step1", self.step1low, "--step2", 'step2.txt', "--wait"])
        br.run(**args)

        out1 = f"{step2prefix}.Y1.regenie"
        out2 = f"{step2prefix}.Y2.regenie"
        expected = "regenie/example/example.test_bin_out_firth_Y1.regenie"

        assert self.read(out1) == read(expected)
        assert not self.exists(out2)

    def test_regenie_phenoList(self):
        step2prefix = f'{self.gcs_output_dir}/phenoColList'
        with open('step2.txt', 'w') as f:
            f.write(f"{self.step2_split}\n--phenoColList Y2\n--out {step2prefix}")

        args = br.parse_input_args(["--step1", self.step1low, "--step2", 'step2.txt', "--wait"])
        br.run(**args)

        out1 = f"{step2prefix}.Y1.regenie"
        out2 = f"{step2prefix}.Y2.regenie"

        assert len(self.read(out2)) > 0
        assert not self.exists(out1)
