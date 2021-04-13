import hailtop.batch as hb

url_root = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased"
chr_i = [str(x) for x in range(1,23)]

backend = hb.ServiceBackend()
batch = hb.Batch(backend=backend, name="1kg-highcov")
for i in chr_i:
    j = batch.new_job(name=i)
    j.image("gcr.io/broad-ctsa/datasets:041421")
    j.command(f"wget -c -O - {url_root}/CCDG_14151_B01_GRM_WGS_2020-08-05_chr{i}.filtered.shapeit2-duohmm-phased.vcf.gz | "
              f"zcat | "
              f"bgzip -c | "
              f"gsutil cp - gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_phased_chr{i}_GRCh38.vcf.bgz")
for i in ["X"]:
    j = batch.new_job(name=i)
    j.image("gcr.io/broad-ctsa/datasets:041421")
    j.command(f"wget -c -O - {url_root}/CCDG_14151_B01_GRM_WGS_2020-08-05_chr{i}.filtered.eagle2-phased.vcf.gz | "
              f"zcat | "
              f"bgzip -c | "
              f"gsutil cp - gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_phased_chr{i}_GRCh38.vcf.bgz")
batch.run(open=True, wait=False)
backend.close()
