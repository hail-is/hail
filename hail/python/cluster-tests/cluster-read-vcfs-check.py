import json
import hail as hl

gvcfs = ['gs://hail-akotlar-syvst/HG00096.g.vcf.gz',
         'gs://hail-akotlar-syvst/HG00268.g.vcf.gz']
hl.init(master='spark://spark-master:7078', default_reference='GRCh38')
parts = [
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
parts_str = json.dumps(parts)
vcfs = hl.import_vcfs(gvcfs, parts_str)
