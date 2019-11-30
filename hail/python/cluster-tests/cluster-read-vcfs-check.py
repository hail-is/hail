import hail as hl

gvcfs = ['gs://hail-common/test-resources/HG00096.g.vcf.gz',
         'gs://hail-common/test-resources/HG00268.g.vcf.gz']
hl.init(default_reference='GRCh38')
parts_json = [
    {'start': {'locus': {'contig': 'chr20', 'position': 17821257}},
     'end': {'locus': {'contig': 'chr20', 'position': 18708366}},
     'includeStart': True,
     'includeEnd': True},
    {'start': {'locus': {'contig': 'chr20', 'position': 18708367}},
     'end': {'locus': {'contig': 'chr20', 'position': 19776611}},
     'includeStart': True,
     'includeEnd': True},
    {'start': {'locus': {'contig': 'chr20', 'position': 19776612}},
     'end': {'locus': {'contig': 'chr20', 'position': 21144633}},
     'includeStart': True,
     'includeEnd': True},
]

parts = hl.tarray(hl.tinterval(hl.tstruct(locus=hl.tlocus('GRCh38'))))._convert_from_json(parts_json)
for mt in hl.import_gvcfs(gvcfs, parts):
    mt._force_count_rows()
