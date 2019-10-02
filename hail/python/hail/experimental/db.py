import requests
import hail as hl

from ..utils.java import Env
from ..typecheck import typecheck_method, oneof
from ..table import table_type
from ..matrixtable import matrix_table_type


class DatasetVersion:
    @staticmethod
    def from_json(doc):
        assert 'url' in doc, doc
        assert 'version' in doc, doc
        return DatasetVersion(doc['url'],
                              doc['version'])

    def __init__(self, url, version):
        self.url = url
        self.version = version

    def maybe_index(self, indexer_key_expr, all_matches):
        return hl.read_table(self.url)._maybe_flexindex_table_by_expr(
            indexer_key_expr, all_matches=all_matches)


class Dataset:
    @staticmethod
    def from_name_and_json(name, doc):
        assert 'description' in doc, doc
        assert 'url' in doc, doc
        assert 'key_properties' in doc, doc
        assert 'versions' in doc, doc
        versions = [DatasetVersion.from_json(x)
                    for x in doc['versions']]
        return Dataset(name,
                       doc['description'],
                       doc['url'],
                       set(doc['key_properties']),
                       versions)

    def __init__(self, name, description, url, key_properties, versions):
        assert set(key_properties).issubset(DB._valid_key_properties)
        self.name = name
        self.description = description
        self.url = url
        self.key_properties = key_properties
        self.versions = versions

    def is_gene_keyed(self):
        return 'gene' in self.key_properties

    def index_compatible_version(self, key_expr):
        all_matches = 'unique' not in self.key_properties
        compatible_indexed_values = [
            index
            for index in (version.maybe_index(key_expr, all_matches)
                          for version in self.versions)
            if index is not None]
        if len(compatible_indexed_values) == 0:
            raise ValueError(
                f'Could not find compatible version of {self.name} for user '
                f'dataset with key {key_expr.dtype}.')
        assert len(compatible_indexed_values) == 1, \
            f'{key_expr.dtype}, {self.name}, {compatible_indexed_values}'
        return compatible_indexed_values[0]


class RowLens:
    def __init__(self, mt):
        assert isinstance(mt, hl.MatrixTable)
        self.mt = mt
        self.key = mt.row_key

    def annotate(self, *args, **kwargs):
        return RowLens(self.mt.annotate_rows(*args, **kwargs))

    def drop(self, *args, **kwargs):
        return RowLens(self.mt.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        return RowLens(self.mt.select_rows(*args, **kwargs))

    def explode(self, *args, **kwargs):
        return RowLens(self.mt.explode_rows(*args, **kwargs))

    def group_by(self, *args, **kwargs):
        return GroupedRowLens(self.mt.group_rows_by(*args, **kwargs))

    def __getitem__(self, *args, **kwargs):
        return self.mt.__getitem__(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.mt.rows().index(*args, **kwargs)

    def unlens(self):
        return self.mt


class GroupedRowLens:
    def __init__(self, mt):
        assert isinstance(mt, hl.GroupedMatrixTable)
        self.mt = mt

    def aggregate(self, *args, **kwargs):
        return RowLens(self.mt.aggregate_rows(*args, **kwargs).result())


class NoLens:
    def __init__(self, t):
        assert isinstance(t, hl.Table)
        self.t = t
        self.key = t.key

    def annotate(self, *args, **kwargs):
        return NoLens(self.t.annotate(*args, **kwargs))

    def drop(self, *args, **kwargs):
        return NoLens(self.t.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        return NoLens(self.t.select(*args, **kwargs))

    def explode(self, *args, **kwargs):
        return NoLens(self.t.explode(*args, **kwargs))

    def group_by(self, *args, **kwargs):
        return GroupedNoLens(self.t.group_by(*args, **kwargs))

    def __getitem__(self, *args, **kwargs):
        return self.t.__getitem__(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.t.index(*args, **kwargs)

    def unlens(self):
        return self.t


class GroupedNoLens:
    def __init__(self, t):
        assert isinstance(t, hl.GroupedTable)
        self.t = t

    def aggregate(self, *args, **kwargs):
        return NoLens(self.t.aggregate(*args, **kwargs))


class DB:
    """An annotation database instance.

    This class facilitates the annotation of genetic datasets with variant
    annotations. It accepts either an HTTP(S) URL to an Annotation DB
    configuration or a python :obj:`dict` describing an Annotation DB
    configuration.

    Examples
    --------
    Create an annotation database connecting to the default Hail Annotation DB:

    >>> db = hl.experimental.DB()
    >>> mt = db.annotate_rows_db(mt, 'gnomad_lof_metrics') # doctest: +SKIP
    """

    _valid_key_properties = {'gene', 'unique'}

    def __init__(self,
                 url=None,
                 json=None):
        if json is not None and url is not None:
            raise ValueError(f'Only specify one of the parameters url and json, '
                             f'received: url={url} and json={json}')
        if json is None:
            if url is None:
                url = ('https://www.googleapis.com/storage/v1/b/hail-common/o/annotationdb%2f' +
                       str(hl.__pip_version__) +
                       '%2fannotation_db.json?alt=media')
            response = requests.get(url)
            json = response.json()
            assert isinstance(json, dict)
        else:
            if not isinstance(json, dict):
                raise ValueError(f'expected a dict mapping dataset names to '
                                 f'configurations, but found {json}')
        self.__by_name = {k: Dataset.from_name_and_json(k, v)
                          for k, v in json.items()}

    def available_databases(self):
        return self._by_name().keys()

    @staticmethod
    def _row_lens(rel):
        if isinstance(rel, hl.MatrixTable):
            return RowLens(rel)
        elif isinstance(rel, hl.Table):
            return NoLens(rel)
        else:
            raise ValueError(
                'annotation database can only annotate Hail MatrixTable or Table')

    def dataset_by_name(self, name):
        if name not in self.__by_name:
            raise ValueError(
                f'{name} not found in annotation database, you may list all '
                f'known dataset names with available_databases()')
        return self.__by_name[name]

    def _annotate_gene_name(self, rel):
        gene_field = Env.get_uid()
        gencode = self.__by_name['gencode'].index_compatible_version(rel.key)
        return gene_field, rel.annotate(**{gene_field: gencode.gene_name})

    @typecheck_method(rel=oneof(table_type, matrix_table_type), names=str)
    def annotate_rows_db(self, rel, *names):
        """Add annotations from datasets specified by name.

        List datasets with at :meth:`.available_databases`. An interactive query
        builder is available in the
        `Hail Annotation Database documentation </docs/0.2/annotation_database_ui.html>`_.

        Examples
        --------
        Annotate a matrix table with the `gnomad_lof_metrics`:

        >>> db = hl.experimental.DB()
        >>> mt = db.annotate_rows_db(mt, 'gnomad_lof_metrics') # doctest: +SKIP

        Annotate a table with `clinvar_gene_summary`, `CADD`, and `DANN`:

        >>> db = hl.experimental.DB()
        >>> mt = db.annotate_rows_db(mt, 'clinvar_gene_summary', 'CADD', 'DANN') # doctest: +SKIP

        Notes
        -----

        If a dataset is gene-keyed, the annotation will be a dictionary mapping
        from gene name to the annotation value. There will be one entry for each
        gene overlapping the given locus.

        If a dataset does not have unique rows for each key (consider the
        `gencode` genes, which may overlap; and `clinvar_variant_summary`, which
        contains many overlapping multiple nucleotide variants), then the result
        will be an array of annotation values, one for each row.

        Parameters
        ----------
        rel : :class:`.MatrixTable` or :class:`.Table`
            The relational object to which to add annotations.
        names : varargs of :obj:`str`
            The names of the datasets with which to annotate `rel`.

        Returns
        -------
        :class:`.MatrixTable` or :class:`.Table
            The original dataset with new annotations added.
        """
        rel = self._row_lens(rel)
        if len(set(names)) != len(names):
            raise ValueError(
                f'cannot annotate same dataset twice, please remove duplicates from: {names}')
        datasets = [self.dataset_by_name(name) for name in names]
        if any(dataset.is_gene_keyed() for dataset in datasets):
            gene_field, rel = self._annotate_gene_name(rel)
        else:
            gene_field = None
        for dataset in datasets:
            if dataset.is_gene_keyed():
                genes = rel.select(gene_field).explode(gene_field)
                genes = genes.annotate(**{
                    dataset.name: dataset.index_compatible_version(genes[gene_field])})
                genes = genes.group_by(*genes.key)\
                             .aggregate(**{
                                 dataset.name: hl.dict(
                                     hl.agg.filter(hl.is_defined(genes[dataset.name]),
                                                   hl.agg.collect((genes[gene_field],
                                                                   genes[dataset.name]))))})
                rel = rel.annotate(**{dataset.name: genes.index(rel.key)[dataset.name]})
            else:
                indexed_value = dataset.index_compatible_version(rel.key)
                if isinstance(indexed_value.dtype, hl.tstruct) and len(indexed_value.dtype) == 0:
                    indexed_value = hl.is_defined(indexed_value)
                rel = rel.annotate(**{dataset.name: indexed_value})
        if gene_field:
            rel = rel.drop(gene_field)
        return rel.unlens()
