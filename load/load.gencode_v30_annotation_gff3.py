import hail as hl

t = hl.import_table('/Users/bfranco/downloads/gencode.v30.annotation.gff3',no_header=True, impute=True, comment=('#'))

t2 = t.annotate(GFF_Columns = t.f8.split(";").map(lambda x: x.split("=")))

t2 = t2.filter(t2.f2 == "gene")

t2 = t2.annotate(interval=hl.interval(hl.locus(t2.f0, t2.f3, 'GRCh38'), hl.locus(t2.f0, t2.f3, 'GRCh38')))

t2 = t2.annotate(GFF_Columns = hl.dict(t2.GFF_Columns.map(lambda x: (x[0], x[1]))))

t2 = t2.annotate(ID=t2.GFF_Columns["ID"], gene_id=t2.GFF_Columns["gene_id"], gene_name=t2.GFF_Columns["gene_name"], gene_type=t2.GFF_Columns["gene_type"], level=t2.GFF_Columns["level"])

t2 = t2.annotate(type=t2.f2, gene_score=t2.f5, gene_strand=t2.f6, gene_phase=t2.f7)

t2 = t2.drop(t2.GFF_Columns, t2.f8, t2.f0, t2.f1, t2.f2, t2.f3, t2.f4, t2.f5, t2.f6, t2.f7)

t2.write('gs://hail-datasets-hail-data/gencode_v30_annotation.mt', overwrite=True)
