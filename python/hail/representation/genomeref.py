from hail.java import handle_py4j, jiterable_to_list
from hail.typecheck import *
from hail.representation.interval import Interval
from hail.utils import wrap_to_list
from hail.history import *


class ReferenceGenome(HistoryMixin):
    """An object that represents a `reference genome <https://en.wikipedia.org/wiki/Reference_genome>`_.

    :param str name: Name of reference.

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

    :param par: List of intervals representing pseudoautosomal regions.
    :type par: list of :class:`.Interval`

    >>> contigs = ["1", "X", "Y", "MT"]
    >>> lengths = {"1": 249250621, "X": 155270560, "Y": 59373566, "MT": 16569}
    >>> par = [Interval.parse("X:60001-2699521")]
    >>> my_ref = ReferenceGenome("my_ref", contigs, lengths, "X", "Y", "MT", par)
    """

    @handle_py4j
    @record_init
    @typecheck_method(name=strlike,
                      contigs=listof(strlike),
                      lengths=dictof(strlike, integral),
                      x_contigs=oneof(strlike, listof(strlike)),
                      y_contigs=oneof(strlike, listof(strlike)),
                      mt_contigs=oneof(strlike, listof(strlike)),
                      par=listof(Interval))
    def __init__(self, name, contigs, lengths, x_contigs=[], y_contigs=[], mt_contigs=[], par=[]):
        contigs = wrap_to_list(contigs)
        x_contigs = wrap_to_list(x_contigs)
        y_contigs = wrap_to_list(y_contigs)
        mt_contigs = wrap_to_list(mt_contigs)
        par_jrep = [interval._jrep for interval in par]

        jrep = (Env.hail().variant.ReferenceGenome
                .apply(name,
                       contigs,
                       lengths,
                       x_contigs,
                       y_contigs,
                       mt_contigs,
                       par_jrep))

        self._init_from_java(jrep)
        self._name = name
        self._contigs = contigs
        self._lengths = lengths
        self._x_contigs = x_contigs
        self._y_contigs = y_contigs
        self._mt_contigs = mt_contigs
        self._par = par

        super(ReferenceGenome, self).__init__()

    @handle_py4j
    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'ReferenceGenome(name=%s, contigs=%s, lengths=%s, x_contigs=%s, y_contigs=%s, mt_contigs=%s, par=%s)' % \
               (self.name, self.contigs, self.lengths, self.x_contigs, self.y_contigs, self.mt_contigs, self.par)

    @handle_py4j
    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    @handle_py4j
    def __hash__(self):
        return self._jrep.hashCode()

    @property
    def name(self):
        """Name of reference genome.

        :rtype: str
        """
        return self._name

    @property
    def contigs(self):
        """Contig names.

        :rtype: list of str
        """
        return self._contigs

    @property
    def lengths(self):
        """Dict of contig name to contig length.

        :rtype: dict of str to int
        """
        return self._lengths

    @property
    def x_contigs(self):
        """X contigs.

        :rtype: list of str
        """
        return self._x_contigs

    @property
    def y_contigs(self):
        """Y contigs.

        :rtype: list of str
        """
        return self._y_contigs

    @property
    def mt_contigs(self):
        """Mitochondrial contigs.

        :rtype: list of str
        """
        return self._mt_contigs

    @property
    def par(self):
        """Pseudoautosomal regions.

        :rtype: list of :class:`.Interval`
        """
        return self._par

    @typecheck_method(contig=strlike)
    def contig_length(self, contig):
        """Contig length.

        :param contig: Contig
        :type contig: str

        :return: Length of contig
        :rtype: int
        """
        return self._jrep.contigLength(contig)

    @classmethod
    @record_classmethod
    @handle_py4j
    def GRCh37(cls):
        """Reference genome for GRCh37.

        Data from `GATK resource bundle <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37.dict>`_.

        >>> grch37 = ReferenceGenome.GRCh37()

        :rtype: :class:`.ReferenceGenome`
        """
        return ReferenceGenome._from_java(Env.hail().variant.ReferenceGenome.GRCh37())

    @classmethod
    @record_classmethod
    @handle_py4j
    def GRCh38(cls):
        """Reference genome for GRCh38.

        Data from `GATK resource bundle <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/Homo_sapiens_assembly38.dict>`_.

        >>> grch38 = ReferenceGenome.GRCh38()

        :rtype: :class:`.ReferenceGenome`
        """
        return ReferenceGenome._from_java(Env.hail().variant.ReferenceGenome.GRCh38())

    @handle_py4j
    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        rg = ReferenceGenome.__new__(cls)
        rg._init_from_java(jrep)
        rg._name = jrep.name()
        rg._contigs = [str(x) for x in jrep.contigs()]
        rg._lengths = {str(x._1()): int(x._2()) for x in jiterable_to_list(jrep.lengths())}
        rg._x_contigs = [str(x) for x in jiterable_to_list(jrep.xContigs())]
        rg._y_contigs = [str(x) for x in jiterable_to_list(jrep.yContigs())]
        rg._mt_contigs = [str(x) for x in jiterable_to_list(jrep.mtContigs())]
        rg._par = [Interval._from_java(x) for x in jrep.par()]
        return rg
