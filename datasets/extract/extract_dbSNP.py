import hailtop.batch as hb

name = "dbSNP"
tmp_bucket = "gs://hail-datasets-tmp"
builds = {
    "GRCh37": {
        "url": "https://ftp.ncbi.nih.gov/snp/latest_release/VCF/GCF_000001405.25.gz",
        "version": "154"
    },
    "GRCh38": {
        "url": "https://ftp.ncbi.nih.gov/snp/latest_release/VCF/GCF_000001405.38.gz",
        "version": "154"
    }
}

backend = hb.ServiceBackend(billing_project="hail-datasets-api")
batch = hb.Batch(backend=backend, name=name)
for build in ["GRCh37", "GRCh38"]:
    vcf = builds[build]["url"]
    version = builds[build]["version"]
    j = batch.new_job(name=f"{name}_{version}_{build}")
    j.image("gcr.io/broad-ctsa/datasets:050521")
    j.command("gcloud -q auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS")
    j.command(f"wget -c -O - {vcf} | "
              "zcat | "
              "bgzip -c | "
              f"gsutil cp - {tmp_bucket}/{name}/{name}_{version}_{build}.vcf.bgz")
batch.run(open=True, wait=False)
backend.close()
