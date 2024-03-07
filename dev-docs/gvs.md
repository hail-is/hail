# Genomic Variant Store

The Genomic Variant Store (GVS) is a BigQuery based system for storing human genome sequences and
performing joint-calling. It roughly resembles the "Scalable Variant Call Representation"
implemented in terms of BigQuery SQL tables. It was developed by DSP. It inspired the "split"
representation (one matrix table for reference data and one for variant data) of the Hail Variant
Dataset.

Our main interaction with GVS is when the produce large callsets. They export their data in Avro
format and then use Hail to import and combine that data into a VDS. They have one variant data
table and one reference data table per 4,000 samples so when they export we get a folder per group
of 4,000 samples. Each folder contains a partitioned dataset ordered by genomic locus; however, they
encode their locus in a 64-bit integer by shifting the contig number into the high bits.

We import each folder of Avro files (using the general purpose JVM Avro reader, which is rather
slow) and convert from its point-wise representation (one record per locus per sample) into a wide
representation (one record per locus with an array of possibly missing sample data). This is called
"sample group import". After import, we massage this data into VDS format and write each one as an
intermediate VDS. Finally, we run the VDS Combiner on these intermediate VDSes, of which there may
be tens or a couple hundred, to produce single final VDS. We then import some variant filtration
annotations from them and add these annotations to the variant data matrix table.

We have proposed using Hail instead of BigQuery for the Azure implementation of GVS:
https://docs.google.com/document/d/1OluN0dEIIKtI2KksFDIC_ZFA4aiCXasZ1OwzagGqtio/edit#heading=h.8ghvgsh8r2db
