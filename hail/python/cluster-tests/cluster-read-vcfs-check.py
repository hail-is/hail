import json
import hail as hl

gvcfs = ['gs://hail-ci/gvcfs/HG00096.g.vcf.gz',
         'gs://hail-ci/gvcfs/HG00268.g.vcf.gz']
hl.init(default_reference='GRCh38')
parts = [{'start': {'locus': {'contig': '20', 'position': 1}},
          'end': {'locus': {'contig': '20', 'position': 13509135}},
          'includeStart': True,
          'includeEnd': True},
         {'start': {'locus': {'contig': '20', 'position': 13509136}},
          'end': {'locus': {'contig': '20', 'position': 16493533}},
          'includeStart': True,
          'includeEnd': True},
         {'start': {'locus': {'contig': '20', 'position': 16493534}},
          'end': {'locus': {'contig': '20', 'position': 20000000}},
          'includeStart': True,
          'includeEnd': True}]
parts_str = json.dumps(parts)
vcfs = hl.import_vcfs(gvcfs, parts_str)
