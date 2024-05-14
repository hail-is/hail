import json
import os
import warnings
from typing import Iterable, List, Optional, Set, Tuple, Union

import hail as hl
import pkg_resources
from hailtop.utils import external_requests_client_session, retry_response_returning_functions

from .lens import MatrixRows, TableRows
from ..expr import StructExpression
from ..matrixtable import MatrixTable, matrix_table_type
from ..table import Table, table_type
from ..typecheck import oneof, typecheck_method
from ..utils.java import Env, info


class DatasetVersion:
    """:class:`DatasetVersion` has two constructors: :func:`.from_json` and
    :func:`.get_region`.

    Parameters
    ----------
    url : :obj:`dict` or :obj:`str`
        Nested dictionary of URLs containing key: value pairs, like
        ``cloud: {region: url}`` if using :func:`.from_json` constructor,
        or a string with the URL from appropriate region if using the
        :func:`.get_region` constructor.
    version : :obj:`str`, optional
        String of dataset version, if not ``None``.
    reference_genome : :obj:`str`, optional
        String of dataset reference genome, if not ``None``.
    """

    @staticmethod
    def from_json(doc: dict, cloud: str) -> Optional['DatasetVersion']:
        """Create :class:`.DatasetVersion` object from dictionary.

        Parameters
        ----------
        doc : :obj:`dict`
            Dictionary containing url and version keys.
            Value for url is a :obj:`dict` containing key: value pairs, like
            ``cloud: {region: url}``.
        cloud : :obj:`str`
            Cloud platform to access dataset, either ``'gcp'`` or ``'aws'``.

        Returns
        -------
        :class:`.DatasetVersion` if available on cloud platform, else ``None``.
        """
        assert 'url' in doc, doc
        assert 'version' in doc, doc
        assert 'reference_genome' in doc, doc
        if cloud in doc['url']:
            return DatasetVersion(doc['url'][cloud], doc['version'], doc['reference_genome'])
        else:
            return None

    @staticmethod
    def get_region(name: str, versions: List['DatasetVersion'], region: str) -> List['DatasetVersion']:
        """Get versions of a :class:`.Dataset` in the specified region, if they
        exist.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.
        versions : :class:`list` of :class:`.DatasetVersion`
            List of DatasetVersion objects where the value for :attr:`.url`
            is a :obj:`dict` containing key: value pairs, like ``region: url``.
        region : :obj:`str`
            Region from which to access data, available regions given in
            :attr:`hail.experimental.DB._valid_regions`.

        Returns
        -------
        available_versions : :class:`list` of :class:`.DatasetVersion`
            List of available versions of a class:`.Dataset` for region.
        """
        available_versions = []
        for version in versions:
            if version.in_region(name, region):
                version.url = version.url[region]
                available_versions.append(version)
        return available_versions

    def __init__(self, url: Union[dict, str], version: Optional[str], reference_genome: Optional[str]):
        self.url = url
        self.version = version
        self.reference_genome = reference_genome

    def in_region(self, name: str, region: str) -> bool:
        """Check if a :class:`.DatasetVersion` object is accessible in the
        desired region.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.
        region : :obj:`str`
            Region from which to access data, available regions given in
            :func:`hail.experimental.DB._valid_regions`.

        Returns
        -------
        valid_region : :obj:`bool`
            Whether or not the dataset exists in the specified region.
        """
        current_version = self.version
        available_regions = [k for k in self.url.keys()]
        valid_region = region in available_regions
        if not valid_region:
            message = (
                f'\nName: {name}\n'
                f'Version: {current_version}\n'
                f'This dataset exists but is not yet available in the'
                f' {region} region bucket.\n'
                f'Dataset is currently available in the'
                f' {", ".join(available_regions)} region bucket(s).\n'
                f'Reach out to the Hail team at https://discuss.hail.is/'
                f' to request this dataset in your region.'
            )
            warnings.warn(message, UserWarning, stacklevel=1)
        return valid_region

    def maybe_index(self, indexer_key_expr: StructExpression, all_matches: bool) -> Optional[StructExpression]:
        """Find the prefix of the given indexer expression that can index the
        :class:`.DatasetVersion`, if it exists.

        Parameters
        ----------
        indexer_key_expr : :class:`StructExpression`
            Row key struct from relational object to be annotated.
        all_matches : :obj:`bool`
            ``True`` if `indexer_key_expr` key is not unique, indicated in
            :attr:`.Dataset.key_properties` for each dataset. If ``True``, value
            of `indexer_key_expr` is array of all matches. If ``False``, there
            will only be single value of expression.

        Returns
        -------
        :class:`StructExpression`, optional
            Struct of compatible indexed values, if they exist.
        """
        return hl.read_table(self.url)._maybe_flexindex_table_by_expr(indexer_key_expr, all_matches=all_matches)


class Dataset:
    """Dataset object constructed from name, description, url, key_properties,
    and versions specified in JSON configuration file or a provided :obj:`dict`
    mapping dataset names to configurations.

    Parameters
    ----------
    name : :obj:`str`
        Name of dataset.
    description : :obj:`str`
        Brief description of dataset.
    url : :obj:`str`
        Cloud URL to access dataset.
    key_properties : :class:`set` of :obj:`str`
        Set containing key property strings, if present. Valid properties
        include ``'gene'`` and ``'unique'``.
    versions : :class:`list` of :class:`.DatasetVersion`
        List of :class:`.DatasetVersion` objects.
    """

    @staticmethod
    def from_name_and_json(name: str, doc: dict, region: str, cloud: str) -> Optional['Dataset']:
        """Create :class:`.Dataset` object from dictionary.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.
        doc : :obj:`dict`
            Dictionary containing dataset description, url, key_properties, and
            versions.
        region : :obj:`str`
            Region from which to access data, available regions given in
            :func:`hail.experimental.DB._valid_regions`.
        cloud : :obj:`str`
            Cloud platform to access dataset, either ``'gcp'`` or ``'aws'``.

        Returns
        -------
        :class:`Dataset`, optional
            If versions exist for region returns a :class:`.Dataset` object,
            else ``None``.
        """
        assert 'annotation_db' in doc, doc
        assert 'key_properties' in doc['annotation_db'], doc['annotation_db']
        assert 'description' in doc, doc
        assert 'url' in doc, doc
        assert 'versions' in doc, doc
        key_properties = set(x for x in doc['annotation_db']['key_properties'] if x is not None)
        versions = [
            DatasetVersion.from_json(x, cloud)
            for x in doc['versions']
            if DatasetVersion.from_json(x, cloud) is not None
        ]
        versions_in_region = DatasetVersion.get_region(name, versions, region)
        if versions_in_region:
            return Dataset(name, doc['description'], doc['url'], key_properties, versions_in_region)

    def __init__(self, name: str, description: str, url: str, key_properties: Set[str], versions: List[DatasetVersion]):
        assert set(key_properties).issubset(DB._valid_key_properties)
        self.name = name
        self.description = description
        self.url = url
        self.key_properties = key_properties
        self.versions = versions

    @property
    def is_gene_keyed(self) -> bool:
        """If a :class:`Dataset` is gene keyed.

        Returns
        -------
        :obj:`bool`
            Whether or not dataset is gene keyed.
        """
        return 'gene' in self.key_properties

    def index_compatible_version(self, key_expr: StructExpression) -> StructExpression:
        """Get index from compatible version of annotation dataset.

        Checks for compatible indexed values from each :class:`.DatasetVersion`
        in :attr:`.Dataset.versions`, where `key_expr` is the row key struct
        from the dataset to be annotated.

        Parameters
        ----------
        key_expr : :class:`.StructExpression`
            Row key struct from relational object to be annotated.

        Returns
        -------
        :class:`.StructExpression`
            Struct of compatible indexed values.
        """
        all_matches = 'unique' not in self.key_properties
        compatible_indexed_values = [
            (version.maybe_index(key_expr, all_matches), version.version)
            for version in self.versions
            if version.maybe_index(key_expr, all_matches) is not None
        ]
        if len(compatible_indexed_values) == 0:
            versions = [f'{(v.version, v.reference_genome)}' for v in self.versions]
            raise ValueError(
                f'Could not find compatible version of {self.name} for user'
                f' dataset with key {key_expr.dtype}.\n'
                f'This annotation dataset is available for the following'
                f' versions and reference genome builds: {", ".join(versions)}.'
            )
        else:
            indexed_values = sorted(compatible_indexed_values, key=lambda x: x[1])[-1]

        if len(compatible_indexed_values) > 1:
            info(
                f'index_compatible_version: More than one compatible version'
                f' exists for annotation dataset: {self.name}. Rows have been'
                f' annotated with version {indexed_values[1]}.'
            )
        return indexed_values[0]


class DB:
    """An annotation database instance.

    This class facilitates the annotation of genetic datasets with variant annotations. It accepts
    either an HTTP(S) URL to an Annotation DB configuration or a Python :obj:`dict` describing an
    Annotation DB configuration. User must specify the `region` (aws: ``'us'``, gcp:
    ``'us-central1'`` or ``'europe-west1'``) in which the cluster is running if connecting to the
    default Hail Annotation DB.  User must also specify the `cloud` platform that they are using
    (``'gcp'`` or ``'aws'``).

    Parameters
    ----------
    region : :obj:`str`
        Region cluster is running in, either ``'us'``, ``'us-central1'``, or ``'europe-west1'``
        (default is ``'us-central1'``).
    cloud : :obj:`str`
        Cloud platform, either ``'gcp'`` or ``'aws'`` (default is ``'gcp'``).
    url : :obj:`str`, optional
        Optional URL to annotation DB configuration, if using custom configuration
        (default is ``None``).
    config : :obj:`str`, optional
        Optional :obj:`dict` describing an annotation DB configuration, if using
        custom configuration (default is ``None``).

    Note
    ----
    The ``'aws'`` `cloud` platform is currently only available for the ``'us'``
    `region`.

    Examples
    --------
    Create an annotation database connecting to the default Hail Annotation DB:

    >>> db = hl.experimental.DB(region='us-central1', cloud='gcp')
    """

    _valid_key_properties = {'gene', 'unique'}
    _valid_regions = {'us', 'us-central1', 'europe-west1'}
    _valid_clouds = {'gcp', 'aws'}
    _valid_combinations = {('us', 'aws'), ('us-central1', 'gcp'), ('europe-west1', 'gcp')}

    def __init__(
        self,
        *,
        region: str = 'us-central1',
        cloud: str = 'gcp',
        url: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        if region not in DB._valid_regions:
            raise ValueError(
                f'Specify valid region parameter,'
                f' received: region={repr(region)}.\n'
                f'Valid regions are {DB._valid_regions}.'
            )
        if cloud not in DB._valid_clouds:
            raise ValueError(
                f'Specify valid cloud parameter,'
                f' received: cloud={repr(cloud)}.\n'
                f'Valid cloud platforms are {DB._valid_clouds}.'
            )
        if (region, cloud) not in DB._valid_combinations:
            raise ValueError(
                f'The {repr(region)} region is not available for'
                f' the {repr(cloud)} cloud platform. '
                f'Valid region, cloud combinations are'
                f' {DB._valid_combinations}.'
            )
        if config is not None and url is not None:
            raise ValueError(
                f'Only specify one of the parameters url and' f' config, received: url={url} and config={config}'
            )
        if config is None:
            if url is None:
                config_path = pkg_resources.resource_filename(__name__, 'datasets.json')
                assert os.path.exists(config_path), f'{config_path} does not exist'
                with open(config_path) as f:
                    config = json.load(f)
            else:
                session = external_requests_client_session()
                response = retry_response_returning_functions(session.get, url)
                config = response.json()
            assert isinstance(config, dict)
        else:
            if not isinstance(config, dict):
                raise ValueError(f'expected a dict mapping dataset names to ' f'configurations, but found {config}')
        config = {k: v for k, v in config.items() if 'annotation_db' in v}
        self.region = region
        self.cloud = cloud
        self.url = url
        self.config = config
        self.__by_name = {
            k: Dataset.from_name_and_json(k, v, region, cloud)
            for k, v in config.items()
            if Dataset.from_name_and_json(k, v, region, cloud) is not None
        }

    @property
    def available_datasets(self) -> List[str]:
        """List of names of available annotation datasets.

        Returns
        -------
        :obj:`list`
            List of available annotation datasets.
        """
        return sorted(self.__by_name.keys())

    @staticmethod
    def _row_lens(rel: Union[Table, MatrixTable]) -> Union[TableRows, MatrixRows]:
        """Get row lens from relational object.

        Parameters
        ----------
        rel : :class:`Table` or :class:`MatrixTable`

        Returns
        -------
        :class:`TableRows` or :class:`MatrixRows`
        """
        if isinstance(rel, MatrixTable):
            return MatrixRows(rel)
        elif isinstance(rel, Table):
            return TableRows(rel)
        else:
            raise ValueError('annotation database can only annotate Hail' ' MatrixTable or Table')

    def _dataset_by_name(self, name: str) -> Dataset:
        """Retrieve :class:`Dataset` object by name.

        Parameters
        ----------
        name : :obj:`str`
            Name of dataset.

        Returns
        -------
        :class:`Dataset`
        """
        if name not in self.__by_name:
            raise ValueError(
                f'{name} not found in annotation database,'
                f' you may list all known dataset names'
                f' with available_datasets'
            )
        return self.__by_name[name]

    def _annotate_gene_name(self, rel: Union[TableRows, MatrixRows]) -> Tuple[str, Union[TableRows, MatrixRows]]:
        """Annotate row lens with gene name if annotation dataset is gene
        keyed.

        Parameters
        ----------
        rel : :class:`TableRows` or :class:`MatrixRows`
            Row lens of relational object to be annotated.

        Returns
        -------
        :class:`tuple`
        """
        gene_field = Env.get_uid()
        gencode = self.__by_name['gencode'].index_compatible_version(rel.key)
        return gene_field, rel.annotate(**{gene_field: gencode.gene_name})

    def _check_availability(self, names: Iterable) -> None:
        """Check if datasets given in `names` are available in the annotation
        database instance.

        Parameters
        ----------
        names : :obj:`iterable`
            Names to check.
        """
        unavailable = [x for x in names if x not in self.__by_name.keys()]
        if unavailable:
            raise ValueError(f'datasets: {unavailable} not available' f' in the {self.region} region.')

    @typecheck_method(rel=oneof(table_type, matrix_table_type), names=str)
    def annotate_rows_db(self, rel: Union[Table, MatrixTable], *names: str) -> Union[Table, MatrixTable]:
        """Add annotations from datasets specified by name to a relational
        object.

        List datasets with :attr:`~.available_datasets`.

        An interactive query builder is available in the
        `Hail Annotation Database documentation
        </docs/0.2/annotation_database_ui.html>`_.

        Examples
        --------
        Annotate a :class:`.MatrixTable` with ``gnomad_lof_metrics``:

        >>> db = hl.experimental.DB(region='us-central1', cloud='gcp')
        >>> mt = db.annotate_rows_db(mt, 'gnomad_lof_metrics') # doctest: +SKIP

        Annotate a :class:`.Table` with ``clinvar_gene_summary``, ``CADD``,
        and ``DANN``:

        >>> db = hl.experimental.DB(region='us-central1', cloud='gcp')
        >>> ht = db.annotate_rows_db(ht, 'clinvar_gene_summary', 'CADD', 'DANN') # doctest: +SKIP

        Notes
        -----

        If a dataset is gene-keyed, the annotation will be a dictionary mapping
        from gene name to the annotation value. There will be one entry for each
        gene overlapping the given locus.

        If a dataset does not have unique rows for each key (consider the
        ``gencode`` genes, which may overlap; and ``clinvar_variant_summary``,
        which contains many overlapping multiple nucleotide variants), then the
        result will be an array of annotation values, one for each row.

        Parameters
        ----------
        rel : :class:`.MatrixTable` or :class:`.Table`
            The relational object to which to add annotations.
        names : varargs of :class:`str`
            The names of the datasets with which to annotate `rel`.

        Returns
        -------
        :class:`.MatrixTable` or :class:`.Table`
            The relational object `rel`, with the annotations from `names`
            added.
        """
        rel = self._row_lens(rel)
        if len(set(names)) != len(names):
            raise ValueError(f'cannot annotate same dataset twice,' f' please remove duplicates from: {names}')
        self._check_availability(names)
        datasets = [self._dataset_by_name(name) for name in names]
        if any(dataset.is_gene_keyed for dataset in datasets):
            gene_field, rel = self._annotate_gene_name(rel)
        else:
            gene_field = None
        for dataset in datasets:
            if dataset.is_gene_keyed:
                genes = rel.select(gene_field).explode(gene_field)
                genes = genes.annotate(**{dataset.name: dataset.index_compatible_version(genes[gene_field])})
                genes = genes.group_by(*genes.key).aggregate(**{
                    dataset.name: hl.dict(
                        hl.agg.filter(
                            hl.is_defined(genes[dataset.name]),
                            hl.agg.collect((genes[gene_field], genes[dataset.name])),
                        )
                    )
                })
                rel = rel.annotate(**{dataset.name: genes.index(rel.key)[dataset.name]})
            else:
                indexed_value = dataset.index_compatible_version(rel.key)
                if isinstance(indexed_value.dtype, hl.tstruct) and len(indexed_value.dtype) == 0:
                    indexed_value = hl.is_defined(indexed_value)
                rel = rel.annotate(**{dataset.name: indexed_value})
        if gene_field:
            rel = rel.drop(gene_field)
        return rel.unlens()
