import hailtop.batch as hb

name = "CADD"
tmp_bucket = "gs://hail-datasets-tmp"
builds = {
    "GRCh37": {
        "snvs_url": "https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh37/whole_genome_SNVs.tsv.gz",
        "indels_url": "https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh37/InDels.tsv.gz",
        "version": "v1.6"
    },
    "GRCh38": {
        "snvs_url": "https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh38/whole_genome_SNVs.tsv.gz",
        "indels_url": "https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh38/gnomad.genomes.r3.0.indel.tsv.gz",
        "version": "v1.6"
    }
}

backend = hb.ServiceBackend(billing_project="hail-datasets-api")
batch = hb.Batch(backend=backend, name=name)
for build in ["GRCh37", "GRCh38"]:
    snvs_url = builds[build]["snvs_url"]
    indels_url = builds[build]["indels_url"]
    version = builds[build]["version"]

    j = batch.new_job(name=f"{name}_{version}_{build}")
    j.image("gcr.io/broad-ctsa/datasets:050521")
    j.command("gcloud -q auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS")
    j.command(f"wget -c -O - {snvs_url} {indels_url} | "
              "zcat | "
              "grep -v '^#' | "
              """awk -v FS=$'\t' -v OFS=$'\t' 'BEGIN {print "chromosome","position","ref","alt","raw_score","PHRED_score"} {print $0}' | """
              "bgzip -c | "
              f"gsutil cp - {tmp_bucket}/{name}/{name}_{version}_{build}.tsv.bgz")
batch.run(open=True, wait=False)
backend.close()
