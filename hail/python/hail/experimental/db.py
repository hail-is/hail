
import requests

class DB:

    _annotation_dataset_urls = None
    
    @staticmethod
    def annotation_dataset_urls():
        if MatrixTable._annotation_dataset_urls is None:
            r = requests.get("http://storage.googleapis.com/hail-datasets/datasets.json")
            j = r.json()
            MatrixTable._annotation_dataset_urls = {(x["name"], x["reference_genome"]): x["path"] for x in j}
            
            return MatrixTable._annotation_dataset_urls

    def annotate_rows_db(self,name):
        """
        Examples
        --------
        Annotates rows based on keyword implementation of annotation name. 
        
        >>> import hail as hl
        >>> annotate_rows_db('vep', 'cadd', '...', name='something_else')
        >>> db = hl.annotation_database(config)
        
        >>> cadd = db.get_dataset('CADD')
        
        >>> ht = db.annotate(ht, 'vep', 'CADD', 'gnomAD')  # adds the vep, CADD, and gnomAD annotations to ht
        ...
        
        Parameters
        ----------
        name: keyword argument of the annotation.
        
        Returns
        -------
        :class:`.MatrixTable`
        """
        d = MatrixTable.annotation_dataset_urls()
        reference_genome = self.row_key.locus.dtype.reference_genome.name
        url = d[(name,reference_genome)]
        t = hl.read_table(url)
                
        return self.annotate_rows(name=t[self.row_key])
