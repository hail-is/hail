import hail as hl

ht_samples = hl.import_table(
        "gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_samples_ped_population.txt.bgz",
        delimiter="\s+",
        impute=True)

ht_samples = ht_samples.annotate(FatherID=hl.if_else(ht_samples.FatherID == "0",
                                                     hl.missing(hl.tstr),
                                                     ht_samples.FatherID),
                                 MotherID=hl.if_else(ht_samples.MotherID == "0",
                                                     hl.missing(hl.tstr),
                                                     ht_samples.MotherID),
                                 Sex=hl.if_else(ht_samples.Sex == 1, "male", "female"))
ht_samples = ht_samples.key_by("SampleID")

n_rows = ht_samples.count()
n_partitions = ht_samples.n_partitions()

ht_samples = ht_samples.annotate_globals(
        metadata=hl.struct(
                name="1000_Genomes_HighCov_samples",
                n_rows=n_rows,
                n_partitions=n_partitions
        )
)

ht_samples.write("gs://hail-datasets-us/1000_Genomes_NYGC_30x_HighCov_samples.ht", overwrite=True)

ht_samples = hl.read_table("gs://hail-datasets-us/1000_Genomes_NYGC_30x_HighCov_samples.ht")
ht_samples.describe()
