import requests
import hail as hl


class DB:
    _annotation_dataset_urls = None
    _geneDict_dataset_urls = None

    @staticmethod
    def annotation_dataset_urls():
        if DB._annotation_dataset_urls is None:
            assert DB._geneDict_dataset_urls is None
            response = requests.get(
                'https://www.googleapis.com/storage/v1/b/'
                'hail-common/o/annotationdb%2f1%2fannotation_db.json?alt=media')
            datasets = response.json()

            DB._annotation_dataset_urls = {
                (dataset["name"], dataset.get("reference_genome")): dataset["path"]
                for dataset in datasets}
            DB._geneDict_dataset_urls = {
                dataset["name"]: dataset["gene_key"]
                for dataset in datasets}

        return DB._annotation_dataset_urls, DB._geneDict_dataset_urls

    def annotate_rows_db(self, mt, *names):
        """
        Examples
        --------
        Annotates rows based on keyword implementation of annotation name.
        The user can type in multiple annotation names when attaching to their datasets.
        >>> db = hl.experimental.DB()
        >>> mt = db.annotate_rows_db(mt, 'gnomad_lof_metrics')

        Parameters
        ----------
        names: Keyword argument of the annotation.
        Can include multiple annotations at one time.

        Returns
        -------
        :class:`.MatrixTable`
        """
        d, geneDict = DB.annotation_dataset_urls()
        reference_genome = mt.row_key.locus.dtype.reference_genome.name
        for name in names:
            gene_key = geneDict[(name)]
            if gene_key is True:
                gene_url = d[('gencode', reference_genome)]
                t = hl.read_table(gene_url)
                mt = mt.annotate_rows(gene_name=t[mt.locus].gene_name)
                url = d[(name, None)]
                t2 = hl.read_table(url)
                mt = mt.annotate_rows(**{name: t2[mt.gene_name]})
                mt = mt.drop('gene_name')
            else:
                url = d[(name, reference_genome)]
                t = hl.read_table(url)
                if len(t.key) > 1:
                    mt = mt.annotate_rows(**{name: t[mt.row_key]})
                else:
                    mt = mt.annotate_rows(**{name: t[mt.locus]})
        return mt
