import hail as hl

ht_samples = hl.read_table("gs://hail-datasets-us/1000_Genomes_NYGC_30x_HighCov_samples.ht")

mt = hl.import_vcf(
        "gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_chrY_GRCh38.vcf.bgz",
        reference_genome="GRCh38")

n_rows, n_cols = mt.count()
n_partitions = mt.n_partitions()

mt = mt.annotate_globals(
        metadata=hl.struct(
                name="1000_Genomes_HighCov_chrY",
                reference_genome="GRCh38",
                n_rows=n_rows,
                n_cols=n_cols,
                n_partitions=n_partitions
        )
)

mt = mt.annotate_cols(**ht_samples[mt.s])
mt = hl.sample_qc(mt)
mt = hl.variant_qc(mt)

mt.write("gs://hail-datasets-us/1000_Genomes_chrY_NYGC_30x_HighCov_GRCh38.mt", overwrite=True)

mt = hl.read_matrix_table("gs://hail-datasets-us/1000_Genomes_chrY_NYGC_30x_HighCov_GRCh38.mt")
mt.describe()
