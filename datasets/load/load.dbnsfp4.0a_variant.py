import hail as hl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Reference genome build to load.')
args = parser.parse_args()

build = args.b

if build == 'GRCh38':
    t = hl.import_table('gs://hail-common/datasets/1/dbnsfp/dbNSFP4.0a_variant.tsv.bgz',force=True, impute=True, missing=".")

    t = t.annotate(
               locus=hl.locus('chr' + t['#chr'], hl.int(t['pos(1-based)']), reference_genome='GRCh38'),
               alleles=[t.ref, t.alt]
                   )
    t = t.key_by(t.locus, t.alleles)


if build == 'GRCh37':
    t = hl.import_table('gs://hail-common/datasets/1/dbnsfp/dbNSFP4.0a_variant.tsv.bgz',force=True, impute=True, missing=".")

    
    t = t.annotate(
                locus=hl.locus(t['#chr'].replace('M', 'MT'), hl.int(t['hg19_pos(1-based)']), reference_genome='GRCh37'),
                alleles=[t.ref, t.alt]
                   )
    t = t.key_by(t.locus, t.alleles)

t = t.transmute(chr = t['#chr'])

t.write(f'gs://hail-common/datasets/1/dbnsfp4.0a.{build}.ht', overwrite=True)

