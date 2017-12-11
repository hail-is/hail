from hail.history import *
from hail.typecheck import *
from hail.utils import wrap_to_list
from hail.utils.java import handle_py4j, jiterable_to_list, Env


class GenomeReference(HistoryMixin):
    """An object that represents a `reference genome <https://en.wikipedia.org/wiki/Reference_genome>`__.

    :param str name: Name of reference. Must be unique and not one of Hail's predefined references "GRCh37" and "GRCh38".

    :param contigs: Contig names.
    :type contigs: list of str

    :param lengths: Dict of contig names to contig lengths.
    :type lengths: dict of str to int

    :param x_contigs: Contigs to be treated as X chromosomes.
    :type x_contigs: str or list of str

    :param y_contigs: Contigs to be treated as Y chromosomes.
    :type y_contigs: str or list of str

    :param mt_contigs: Contigs to be treated as mitochondrial DNA.
    :type mt_contigs: str or list of str

    :param par: List of tuples with (contig, start, end)
    :type par: list of tuple of (str or int, int, int)

    >>> contigs = ["1", "X", "Y", "MT"]
    >>> lengths = {"1": 249250621, "X": 155270560, "Y": 59373566, "MT": 16569}
    >>> par = [("X", 60001, 2699521)]
    >>> my_ref = GenomeReference("my_ref", contigs, lengths, "X", "Y", "MT", par)
    """

    @handle_py4j
    @record_init
    @typecheck_method(name=strlike,
                      contigs=listof(strlike),
                      lengths=dictof(strlike, integral),
                      x_contigs=oneof(strlike, listof(strlike)),
                      y_contigs=oneof(strlike, listof(strlike)),
                      mt_contigs=oneof(strlike, listof(strlike)),
                      par=listof(sized_tupleof(oneof(strlike, integral), integral, integral)))
    def __init__(self, name, contigs, lengths, x_contigs=[], y_contigs=[], mt_contigs=[], par=[]):
        contigs = wrap_to_list(contigs)
        x_contigs = wrap_to_list(x_contigs)
        y_contigs = wrap_to_list(y_contigs)
        mt_contigs = wrap_to_list(mt_contigs)

        par_strings = ["{}:{}-{}".format(contig, start, end) for (contig, start, end) in par]

        jrep = (Env.hail().variant.GenomeReference
                .apply(name,
                       contigs,
                       lengths,
                       x_contigs,
                       y_contigs,
                       mt_contigs,
                       par_strings))

        self._init_from_java(jrep)
        self._name = name
        self._contigs = contigs
        self._lengths = lengths
        self._x_contigs = x_contigs
        self._y_contigs = y_contigs
        self._mt_contigs = mt_contigs
        self._par = None
        self._par_tuple = par

        super(GenomeReference, self).__init__()

    def _set_history(self, history):
        if not self._history_was_set: # only set varid if GenomeReference was constructed by user
            self._history = history.set_varid(self.name)
            self._history_was_set = True
        else:
            super(GenomeReference, self)._set_history(history)

    @handle_py4j
    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        if not self._par_tuple:
            self._par_tuple = [(x.start.contig, x.start.position, x.end.position) for x in self.par]
        return 'GenomeReference(name=%s, contigs=%s, lengths=%s, x_contigs=%s, y_contigs=%s, mt_contigs=%s, par=%s)' % \
               (self.name, self.contigs, self.lengths, self.x_contigs, self.y_contigs, self.mt_contigs, self._par_tuple)

    @handle_py4j
    def __eq__(self, other):
        return isinstance(other, GenomeReference) and self._jrep.equals(other._jrep)

    @handle_py4j
    def __hash__(self):
        return self._jrep.hashCode()

    @property
    def name(self):
        """Name of genome reference.

        :rtype: str
        """
        if self._name is None:
            self._name = self._jrep.name()
        return self._name

    @property
    def contigs(self):
        """Contig names.

        :rtype: list of str
        """
        if self._contigs is None:
            self._contigs = [str(x) for x in self._jrep.contigs()]
        return self._contigs

    @property
    def lengths(self):
        """Dict of contig name to contig length.

        :rtype: dict of str to int
        """
        if self._lengths is None:
            self._lengths = {str(x._1()): int(x._2()) for x in jiterable_to_list(self._jrep.lengths())}
        return self._lengths

    @property
    def x_contigs(self):
        """X contigs.

        :rtype: list of str
        """
        if self._x_contigs is None:
            self._x_contigs = [str(x) for x in jiterable_to_list(self._jrep.xContigs())]
        return self._x_contigs

    @property
    def y_contigs(self):
        """Y contigs.

        :rtype: list of str
        """
        if self._y_contigs is None:
            self._y_contigs = [str(x) for x in jiterable_to_list(self._jrep.yContigs())]
        return self._y_contigs

    @property
    def mt_contigs(self):
        """Mitochondrial contigs.

        :rtype: list of str
        """
        if self._mt_contigs is None:
            self._mt_contigs = [str(x) for x in jiterable_to_list(self._jrep.mtContigs())]
        return self._mt_contigs

    @property
    def par(self):
        """Pseudoautosomal regions.

        :rtype: list of :class:`.Interval`
        """

        from hail.genetics.interval import Interval
        if self._par is None:
            self._par = [Interval._from_java(jrep, self) for jrep in self._jrep.par()]
        return self._par

    @typecheck_method(contig=strlike)
    def contig_length(self, contig):
        """Contig length.

        :param contig: Contig
        :type contig: str

        :return: Length of contig.
        :rtype: int
        """
        if contig in self.lengths:
            return self.lengths[contig]
        else:
            raise KeyError("Contig `{}' is not in reference genome.".format(contig))

    @classmethod
    @record_classmethod
    @handle_py4j
    def GRCh37(cls):
        """Reference genome for GRCh37.

        Data from `GATK resource bundle <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37.dict>`__.

        >>> grch37 = GenomeReference.GRCh37()

        :rtype: :class:`.GenomeReference`
        """
        return GenomeReference._from_java(Env.hail().variant.GenomeReference.GRCh37())

    @classmethod
    @record_classmethod
    @handle_py4j
    def GRCh38(cls):
        """Reference genome for GRCh38.

        Data from `GATK resource bundle <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/Homo_sapiens_assembly38.dict>`__.

        >>> grch38 = GenomeReference.GRCh38()

        :rtype: :class:`.GenomeReference`
        """
        return GenomeReference._from_java(Env.hail().variant.GenomeReference.GRCh38())

    @classmethod
    @record_classmethod
    @handle_py4j
    @typecheck_method(file=strlike)
    def read(cls, file):
        """Load reference genome from a JSON file.

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

        **Notes**

        `name` must be unique and not overlap with Hail's pre-instantiated references: "GRCh37" and "GRCh38".
        The contig names in `xContigs`, `yContigs`, and `mtContigs` must be present in `contigs`. The intervals listed in
        `par` must have contigs in either `xContigs` or `yContigs` and have positions between 0 and the contig length given
        in `contigs`.

        :param file: Path to JSON file.
        :type file: str

        :rtype: :class:`.GenomeReference`
        """
        return GenomeReference._from_java(Env.hail().variant.GenomeReference.fromFile(Env.hc()._jhc, file))

    @handle_py4j
    @typecheck_method(output=strlike)
    def write(self, otuput):
        """"Write this reference genome to a file in JSON format.

        **Examples**

        >>> my_gr = GenomeReference("new_reference", ["x", "y", "z"], {"x": 500, "y": 300, "z": 200})
        >>> my_gr.write("output/new_reference.json")

        **Notes**

        Use :py:class:`~hail.GenomeReference.read` to reimport the exported reference genome
        in a new HailContext session.

        :param str output: Path of JSON file to write.
        """

        self._jrep.write(Env.hc()._jhc, file)

    @handle_py4j
    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        gr = GenomeReference.__new__(cls)
        gr._init_from_java(jrep)
        gr._name = None
        gr._contigs = None
        gr._lengths = None
        gr._x_contigs = None
        gr._y_contigs = None
        gr._mt_contigs = None
        gr._par = None
        gr._par_tuple = None
        super(GenomeReference, gr).__init__()
        return gr
