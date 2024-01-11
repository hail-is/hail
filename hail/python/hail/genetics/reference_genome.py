from bisect import bisect_right
import json
import re
from hail.typecheck import typecheck_method, sequenceof, dictof, oneof, sized_tupleof, nullable, transformed, lazy
from hail.utils.misc import wrap_to_list
from hail.utils.java import Env
import hail as hl

rg_type = lazy()
reference_genome_type = oneof(transformed((str, lambda x: hl.get_reference(x))), rg_type)


class ReferenceGenome:
    """An object that represents a `reference genome <https://en.wikipedia.org/wiki/Reference_genome>`__.

    Examples
    --------

    >>> contigs = ["1", "X", "Y", "MT"]
    >>> lengths = {"1": 249250621, "X": 155270560, "Y": 59373566, "MT": 16569}
    >>> par = [("X", 60001, 2699521)]
    >>> my_ref = hl.ReferenceGenome("my_ref", contigs, lengths, "X", "Y", "MT", par)

    Notes
    -----
    Hail comes with predefined reference genomes (case sensitive!):

     - GRCh37, Genome Reference Consortium Human Build 37
     - GRCh38, Genome Reference Consortium Human Build 38
     - GRCm38, Genome Reference Consortium Mouse Build 38
     - CanFam3, Canis lupus familiaris (dog)

    You can access these reference genome objects using :func:`~hail.get_reference`:

    >>> rg = hl.get_reference('GRCh37')
    >>> rg = hl.get_reference('GRCh38')
    >>> rg = hl.get_reference('GRCm38')
    >>> rg = hl.get_reference('CanFam3')

    Note that constructing a new reference genome, either by using the class
    constructor or by using `read` will add the reference genome to the list of
    known references; it is possible to access the reference genome using
    :func:`~hail.get_reference` anytime afterwards.

    Note
    ----
    Reference genome names must be unique. It is not possible to overwrite the
    built-in reference genomes.

    Note
    ----
    Hail allows setting a default reference so that the ``reference_genome``
    argument of :func:`~hail.methods.import_vcf` does not need to be used
    constantly. It is a current limitation of Hail that a custom reference
    genome cannot be used as the ``default_reference`` argument of
    :func:`~hail.init`. In order to set a custom reference genome as default,
    pass the reference as an argument to :func:`~hail.default_reference` after
    initializing Hail.

    Parameters
    ----------
    name : :class:`str`
        Name of reference. Must be unique and NOT one of Hail's
        predefined references: ``'GRCh37'``, ``'GRCh38'``, ``'GRCm38'``,
        ``'CanFam3'`` and ``'default'``.
    contigs : :obj:`list` of :class:`str`
        Contig names.
    lengths : :obj:`dict` of :class:`str` to :obj:`int`
        Dict of contig names to contig lengths.
    x_contigs : :class:`str` or :obj:`list` of :obj:`str`
        Contigs to be treated as X chromosomes.
    y_contigs : :class:`str` or :obj:`list` of :obj:`str`
        Contigs to be treated as Y chromosomes.
    mt_contigs : :class:`str` or :obj:`list` of :obj:`str`
        Contigs to be treated as mitochondrial DNA.
    par : :obj:`list` of :obj:`tuple` of (str, int, int)
        List of tuples with (contig, start, end)
    """

    @classmethod
    def _from_config(cls, config, _builtin=False):
        def par_tuple(p):
            assert p['start']['contig'] == p['end']['contig']
            return (p['start']['contig'], p['start']['position'], p['end']['position'])

        contigs = config['contigs']
        return ReferenceGenome(
            config['name'],
            [c['name'] for c in contigs],
            {c['name']: c['length'] for c in contigs},
            config['xContigs'],
            config['yContigs'],
            config['mtContigs'],
            [par_tuple(p) for p in config['par']],
            _builtin,
        )

    @typecheck_method(
        name=str,
        contigs=sequenceof(str),
        lengths=dictof(str, int),
        x_contigs=oneof(str, sequenceof(str)),
        y_contigs=oneof(str, sequenceof(str)),
        mt_contigs=oneof(str, sequenceof(str)),
        par=sequenceof(sized_tupleof(str, int, int)),
        _builtin=bool,
    )
    def __init__(self, name, contigs, lengths, x_contigs=[], y_contigs=[], mt_contigs=[], par=[], _builtin=False):
        contigs = wrap_to_list(contigs)
        x_contigs = wrap_to_list(x_contigs)
        y_contigs = wrap_to_list(y_contigs)
        mt_contigs = wrap_to_list(mt_contigs)

        self._config = {
            'name': name,
            'contigs': [{'name': c, 'length': l} for c, l in lengths.items()],
            'xContigs': x_contigs,
            'yContigs': y_contigs,
            'mtContigs': mt_contigs,
            'par': [{'start': {'contig': c, 'position': s}, 'end': {'contig': c, 'position': e}} for (c, s, e) in par],
        }

        self._contigs = contigs
        self._lengths = lengths
        self._par_tuple = par
        self._par = [hl.Interval(hl.Locus(c, s, self), hl.Locus(c, e, self)) for (c, s, e) in par]
        self._global_positions = None
        self._global_positions_list = None

        if not _builtin:
            Env.backend().add_reference(self)

        self._sequence_files = None
        self._liftovers = dict()

    def __str__(self):
        return self._config['name']

    def __repr__(self):
        return 'ReferenceGenome(name=%s, contigs=%s, lengths=%s, x_contigs=%s, y_contigs=%s, mt_contigs=%s, par=%s)' % (
            self.name,
            self.contigs,
            self.lengths,
            self.x_contigs,
            self.y_contigs,
            self.mt_contigs,
            self._par_tuple,
        )

    def __eq__(self, other):
        return isinstance(other, ReferenceGenome) and self._config == other._config

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self):
        """Name of reference genome.

        Returns
        -------
        :class:`str`
        """
        return self._config['name']

    @property
    def contigs(self):
        """Contig names.

        Returns
        -------
        :obj:`list` of :class:`str`
        """
        return self._contigs

    @property
    def lengths(self):
        """Dict of contig name to contig length.

        Returns
        -------
        :obj:`dict` of :class:`str` to :obj:`int`
        """
        return self._lengths

    @property
    def x_contigs(self):
        """X contigs.

        Returns
        -------
        :obj:`list` of :class:`str`
        """
        return self._config['xContigs']

    @property
    def y_contigs(self):
        """Y contigs.

        Returns
        -------
        :obj:`list` of :class:`str`
        """
        return self._config['yContigs']

    @property
    def mt_contigs(self):
        """Mitochondrial contigs.

        Returns
        -------
        :obj:`list` of :class:`str`
        """
        return self._config['mtContigs']

    @property
    def par(self):
        """Pseudoautosomal regions.

        Returns
        -------
        :obj:`list` of :class:`.Interval`
        """

        return self._par

    @typecheck_method(contig=str)
    def contig_length(self, contig):
        """Contig length.

        Parameters
        ----------
        contig : :class:`str`
            Contig name.

        Returns
        -------
        :obj:`int`
            Length of contig.
        """
        if contig in self.lengths:
            return self.lengths[contig]
        else:
            raise KeyError("Contig `{}' is not in reference genome.".format(contig))

    @property
    def global_positions_dict(self):
        """Get a dictionary mapping contig names to their global genomic positions.

        Returns
        -------
        :class:`dict`
            A dictionary of contig names to global genomic positions.
        """
        if self._global_positions is None:
            gp = {}
            lengths = self._lengths
            x = 0
            for c in self.contigs:
                gp[c] = x
                x += lengths[c]
            self._global_positions = gp
        return self._global_positions

    @typecheck_method(contig=str)
    def _contig_global_position(self, contig):
        return self.global_positions_dict[contig]

    @classmethod
    @typecheck_method(path=str)
    def read(cls, path):
        """Load reference genome from a JSON file.

        Notes
        -----

        The JSON file must have the following format:

        .. code-block:: text

            {"name": "my_reference_genome",
             "contigs": [{"name": "1", "length": 10000000},
                         {"name": "2", "length": 20000000},
                         {"name": "X", "length": 19856300},
                         {"name": "Y", "length": 78140000},
                         {"name": "MT", "length": 532}],
             "xContigs": ["X"],
             "yContigs": ["Y"],
             "mtContigs": ["MT"],
             "par": [{"start": {"contig": "X","position": 60001},"end": {"contig": "X","position": 2699521}},
                     {"start": {"contig": "Y","position": 10001},"end": {"contig": "Y","position": 2649521}}]
            }


        `name` must be unique and not overlap with Hail's pre-instantiated
        references: ``'GRCh37'``, ``'GRCh38'``, ``'GRCm38'``, ``'CanFam3'``, and
        ``'default'``.
        The contig names in `xContigs`, `yContigs`, and `mtContigs` must be
        present in `contigs`. The intervals listed in `par` must have contigs in
        either `xContigs` or `yContigs` and must have positions between 0 and
        the contig length given in `contigs`.

        Parameters
        ----------
        path : :class:`str`
            Path to JSON file.

        Returns
        -------
        :class:`.ReferenceGenome`
        """
        with hl.hadoop_open(path) as f:
            return ReferenceGenome._from_config(json.load(f))

    @typecheck_method(output=str)
    def write(self, output):
        """ "Write this reference genome to a file in JSON format.

        Examples
        --------

        >>> my_rg = hl.ReferenceGenome("new_reference", ["x", "y", "z"], {"x": 500, "y": 300, "z": 200})
        >>> my_rg.write(f"output/new_reference.json")

        Notes
        -----

        Use :meth:`~hail.genetics.ReferenceGenome.read` to reimport the exported
        reference genome in a new HailContext session.

        Parameters
        ----------
        output : :class:`str`
            Path of JSON file to write.
        """
        with hl.utils.hadoop_open(output, 'w') as f:
            json.dump(self._config, f)

    @typecheck_method(fasta_file=str, index_file=nullable(str))
    def add_sequence(self, fasta_file, index_file=None):
        """Load the reference sequence from a FASTA file.

        Examples
        --------
        Access the GRCh37 reference genome using :func:`~hail.get_reference`:

        >>> rg = hl.get_reference('GRCh37') # doctest: +SKIP

        Add a sequence file:

        >>> rg.add_sequence('gs://hail-common/references/human_g1k_v37.fasta.gz',
        ...                 'gs://hail-common/references/human_g1k_v37.fasta.fai') # doctest: +SKIP

        Add a sequence file with the default index location:

        >>> rg.add_sequence('gs://hail-common/references/human_g1k_v37.fasta.gz') # doctest: +SKIP


        Notes
        -----
        This method can only be run once per reference genome. Use
        :meth:`~has_sequence` to test whether a sequence is loaded.

        FASTA and index files are hosted on google cloud for some of Hail's built-in
        references:

        **GRCh37**

        - FASTA file: ``gs://hail-common/references/human_g1k_v37.fasta.gz``
        - Index file: ``gs://hail-common/references/human_g1k_v37.fasta.fai``

        **GRCh38**

        - FASTA file: ``gs://hail-common/references/Homo_sapiens_assembly38.fasta.gz``
        - Index file: ``gs://hail-common/references/Homo_sapiens_assembly38.fasta.fai``

        Public download links are available
        `here <https://console.cloud.google.com/storage/browser/hail-common/references/>`__.

        Parameters
        ----------
        fasta_file : :class:`str`
            Path to FASTA file. Can be compressed (GZIP) or uncompressed.
        index_file : :obj:`None` or :class:`str`
            Path to FASTA index file. Must be uncompressed. If `None`, replace
            the fasta_file's extension with `fai`.
        """
        if index_file is None:
            index_file = re.sub(r'\.[^.]*$', '.fai', fasta_file)
        Env.backend().add_sequence(self.name, fasta_file, index_file)
        self._sequence_files = (fasta_file, index_file)

    def has_sequence(self):
        """True if the reference sequence has been loaded.

        Returns
        -------
        :obj:`bool`
        """
        return self._sequence_files is not None

    def remove_sequence(self):
        """Remove the reference sequence."""
        self._sequence_files = None
        Env.backend().remove_sequence(self.name)

    @classmethod
    @typecheck_method(
        name=str,
        fasta_file=str,
        index_file=str,
        x_contigs=oneof(str, sequenceof(str)),
        y_contigs=oneof(str, sequenceof(str)),
        mt_contigs=oneof(str, sequenceof(str)),
        par=sequenceof(sized_tupleof(str, int, int)),
    )
    def from_fasta_file(cls, name, fasta_file, index_file, x_contigs=[], y_contigs=[], mt_contigs=[], par=[]):
        """Create reference genome from a FASTA file.

        Parameters
        ----------
        name: :class:`str`
            Name for new reference genome.
        fasta_file : :class:`str`
            Path to FASTA file. Can be compressed (GZIP) or uncompressed.
        index_file : :class:`str`
            Path to FASTA index file. Must be uncompressed.
        x_contigs : :class:`str` or :obj:`list` of :obj:`str`
            Contigs to be treated as X chromosomes.
        y_contigs : :class:`str` or :obj:`list` of :obj:`str`
            Contigs to be treated as Y chromosomes.
        mt_contigs : :class:`str` or :obj:`list` of :obj:`str`
            Contigs to be treated as mitochondrial DNA.
        par : :obj:`list` of :obj:`tuple` of (str, int, int)
            List of tuples with (contig, start, end)

        Returns
        -------
        :class:`.ReferenceGenome`
        """
        par_strings = ["{}:{}-{}".format(contig, start, end) for (contig, start, end) in par]
        config = Env.backend().from_fasta_file(
            name, fasta_file, index_file, x_contigs, y_contigs, mt_contigs, par_strings
        )

        rg = ReferenceGenome._from_config(config)
        rg.add_sequence(fasta_file, index_file)
        return rg

    @typecheck_method(dest_reference_genome=reference_genome_type)
    def has_liftover(self, dest_reference_genome):
        """``True`` if a liftover chain file is available from this reference
        genome to the destination reference.

        Parameters
        ----------
        dest_reference_genome : :class:`str` or :class:`.ReferenceGenome`

        Returns
        -------
        :obj:`bool`
        """
        return dest_reference_genome.name in self._liftovers

    @typecheck_method(dest_reference_genome=reference_genome_type)
    def remove_liftover(self, dest_reference_genome):
        """Remove liftover to `dest_reference_genome`.

        Parameters
        ----------
        dest_reference_genome : :class:`str` or :class:`.ReferenceGenome`
        """
        if dest_reference_genome.name in self._liftovers:
            del self._liftovers[dest_reference_genome.name]
            Env.backend().remove_liftover(self.name, dest_reference_genome.name)

    @typecheck_method(chain_file=str, dest_reference_genome=reference_genome_type)
    def add_liftover(self, chain_file, dest_reference_genome):
        """Register a chain file for liftover.

        Examples
        --------
        Access GRCh37 and GRCh38 using :func:`~hail.get_reference`:

        >>> rg37 = hl.get_reference('GRCh37') # doctest: +SKIP
        >>> rg38 = hl.get_reference('GRCh38') # doctest: +SKIP

        Add a chain file from 37 to 38:

        >>> rg37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', rg38) # doctest: +SKIP

        Notes
        -----
        This method can only be run once per reference genome. Use
        :meth:`~has_liftover` to test whether a chain file has been registered.

        The chain file format is described
        `here <https://genome.ucsc.edu/goldenpath/help/chain.html>`__.

        Chain files are hosted on google cloud for some of Hail's built-in
        references:

        **GRCh37 to GRCh38**
        gs://hail-common/references/grch37_to_grch38.over.chain.gz

        **GRCh38 to GRCh37**
        gs://hail-common/references/grch38_to_grch37.over.chain.gz

        Public download links are available
        `here <https://console.cloud.google.com/storage/browser/hail-common/references/>`__.

        Parameters
        ----------
        chain_file : :class:`str`
            Path to chain file. Can be compressed (GZIP) or uncompressed.
        dest_reference_genome : :class:`str` or :class:`.ReferenceGenome`
            Reference genome to convert to.
        """

        Env.backend().add_liftover(self.name, chain_file, dest_reference_genome.name)
        if dest_reference_genome.name in self._liftovers:
            raise KeyError(f"Liftover already exists from {self.name} to {dest_reference_genome.name}.")
        if dest_reference_genome.name == self.name:
            raise ValueError(f'Destination reference genome cannot have the same name as this reference {self.name}.')
        self._liftovers[dest_reference_genome.name] = chain_file

    @typecheck_method(global_pos=int)
    def locus_from_global_position(self, global_pos: int) -> 'hl.Locus':
        """ "
        Constructs a locus from a global position in reference genome.
        The inverse of :meth:`.Locus.position`.

        Examples
        --------
        >>> rg = hl.get_reference('GRCh37')
        >>> rg.locus_from_global_position(0)
        Locus(contig=1, position=1, reference_genome=GRCh37)

        >>> rg.locus_from_global_position(2824183054)
        Locus(contig=21, position=42584230, reference_genome=GRCh37)

        >>> rg = hl.get_reference('GRCh38')
        >>> rg.locus_from_global_position(2824183054)
        Locus(contig=chr22, position=1, reference_genome=GRCh38)

        Parameters
        ----------
        global_pos : int
            Zero-based global base position along the reference genome.

        Returns
        -------
        :class:`.Locus`
        """
        if global_pos < 0:
            raise ValueError(f"global_pos must be non-negative, got {global_pos}")

        if self._global_positions_list is None:
            # dicts are in insertion order as of 3.7
            self._global_positions_list = list(self.global_positions_dict.values())

        global_positions = self._global_positions_list
        contig = self.contigs[bisect_right(global_positions, global_pos) - 1]
        contig_pos = self.global_positions_dict[contig]

        if global_pos >= contig_pos + self.lengths[contig]:
            raise ValueError(f"global_pos {global_pos} exceeds length of reference genome {self}.")

        return hl.Locus(contig, global_pos - contig_pos + 1, self)


rg_type.set(ReferenceGenome)
