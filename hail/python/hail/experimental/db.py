from ..matrixtable import MatrixTable
import requests
import json
import hail as hl

class DB:
    
    _annotation_dataset_urls = None
    
    @staticmethod
    def annotation_dataset_urls():
        if DB._annotation_dataset_urls is None:
            with open('/Users/bfranco/hail_datasets/hail-datasets/annotation_db.json') as f:
                j = json.loads(f.read())
            
            DB._annotation_dataset_urls = {(x["name"], x["reference_genome"]): (x["path"], x["gene_key"]) for x in j}
        
        return DB._annotation_dataset_urls
    
    def annotate_rows_db(self,mt,*names):
            """
            Examples
            --------
            Annotates rows based on keyword implementation of annotation name. The user can type in multiple annotation names when attaching to their datasets.
            
            >>> import hail as hl
            >>> annotate_rows_db(mt,'DANN','CADD', '...', name='something_else')
            >>> db = hl.annotation_database(config)
            
            
            >>> mt = db.annotate_rows_db(mt, 'vep', 'CADD', 'gnomAD')  # adds the vep, CADD, and gnomAD annotations to mt
            ...
            
            Parameters
            ----------
            names: keyword argument of the annotation. Can include multiple annotations at one time.
            
            Returns
            -------
            :class:`.MatrixTable`
            """
        d = DB.annotation_dataset_urls()
        reference_genome = mt.row_key.locus.dtype.reference_genome.name
        for name in names:
            url, gene_key = d[(name,reference_genome)]
            if gene_key is True:
                gene_url, _ = d[('gencode', reference_genome)]
                t = hl.read_table(gene_url)
                mt = mt.annotate_rows(gene_name=t[mt.row_key].gene_name)
                mt = mt.annotate_rows(**{name:t[mt.gene_name]})
                mt = mt.drop('gene_name')
            else:
                t = hl.read_table(url)
                mt = mt.annotate_rows(**{name:t[mt.row_key]})
        return mt



