import hailtop.batch as hb

phased_url_root = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/" \
                  "1000G_2504_high_coverage/working/20201028_3202_phased"
gt_url_root = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/" \
              "1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot"

backend = hb.ServiceBackend(billing_project="hail-datasets-api")
batch = hb.Batch(backend=backend, name="1kg-highcov")
for i in [str(x) for x in range(1,23)]:
    j = batch.new_job(name=i)
    j.image("gcr.io/broad-ctsa/datasets:041421")
    j.command(f"wget -c -O - {phased_url_root}/CCDG_14151_B01_GRM_WGS_2020-08-05_chr{i}.filtered.shapeit2-duohmm-phased.vcf.gz | "
              f"zcat | "
              f"bgzip -c | "
              f"gsutil cp - gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_phased_chr{i}_GRCh38.vcf.bgz")
for i in ["X"]:
    j = batch.new_job(name=i)
    j.image("gcr.io/broad-ctsa/datasets:041421")
    j.command(f"wget -c -O - {phased_url_root}/CCDG_14151_B01_GRM_WGS_2020-08-05_chr{i}.filtered.eagle2-phased.vcf.gz | "
              f"zcat | "
              f"bgzip -c | "
              f"gsutil cp - gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_phased_chr{i}_GRCh38.vcf.bgz")
for i in ["Y"]:
    j = batch.new_job(name=i)
    j.image("gcr.io/broad-ctsa/datasets:041421")
    j.command(f"wget -c -O - {gt_url_root}/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr{i}.recalibrated_variants.vcf.gz | "
              f"zcat | "
              f"bgzip -c | "
              f"gsutil cp - gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_chr{i}_GRCh38.vcf.bgz")
batch.run(open=True, wait=False)
backend.close()
