from hail.java import Env, handle_py4j, jiterable_to_list
from hail.typecheck import *
from hail.representation.interval import Interval
from hail.representation.variant import Contig
from hail.utils import wrap_to_list


class GenomeReference(object):
    """An object that represents a genome reference.

    :param str name: Name of reference

    :param contigs: List of contigs
    :type contigs: list of :class:`.Contig`

    :param x_contig_names: Names of contigs to be treated as X chromosomes
    :type x_contig_names: str or list of str

    :param y_contig_names: Names of contigs to be treated as Y chromosomes
    :type y_contig_names: str or list of str

    :param mt_contig_names: Names of contigs to be treated as Mitochondrial chromosomes
    :type mt_contig_names: str or list of str

    :param par: List of intervals representing pseudoautosomal regions
    :type par: list of :class:`.Interval`

    >>> contigs = [Contig(1, 249250621), Contig("X", 155270560),
    ...            Contig("Y", 59373566), Contig("MT", 16569)]
    >>> par = [Interval.parse("X:60001-2699521")]
    >>> my_gr = GenomeReference("my_gr", contigs, "X", "Y", "MT", par)
    """

    @handle_py4j
    @typecheck_method(name=strlike,
                      contigs=listof(Contig),
                      x_contig_names=oneof(strlike, listof(strlike)),
                      y_contig_names=oneof(strlike, listof(strlike)),
                      mt_contig_names=oneof(strlike, listof(strlike)),
                      par=listof(Interval))
    def __init__(self, name, contigs, x_contig_names=[], y_contig_names=[], mt_contig_names=[], par=[]):
        contigs_jrep = [c._jrep for c in contigs]
        par_jrep = [interval._jrep for interval in par]
        x_contig_names = wrap_to_list(x_contig_names)
        y_contig_names = wrap_to_list(y_contig_names)
        mt_contig_names = wrap_to_list(mt_contig_names)

        jrep = (Env.hail().variant.GenomeReference
                .apply(name,
                       contigs_jrep,
                       x_contig_names,
                       y_contig_names,
                       mt_contig_names,
                       par_jrep))

        self._init_from_java(jrep)
        self._name = name
        self._contigs = contigs
        self._x_contig_names = x_contig_names
        self._y_contig_names = y_contig_names
        self._mt_contig_names = mt_contig_names
        self._par = par

        Env.hail().variant.GenomeReference.addReference(jrep)

    @handle_py4j
    def __str__(self):
        return self._jrep.toString()

    def __repr__(self):
        return 'GenomeReference(name=%s, contigs=%s, x_contig_names=%s, y_contig_names=%s, mt_contig_names=%s, par=%s)' % (self.name, self.contigs, self.x_contigs, self.y_contigs, self.mt_contigs, self.par)

    @handle_py4j
    def __eq__(self, other):
        return self._jrep.equals(other._jrep)

    @handle_py4j
    def __hash__(self):
        return self._jrep.hashCode()

    @property
    def name(self):
        """Name of genome reference

        :rtype: str
        """
        return self._name

    @property
    def contigs(self):
        """Contigs

        :rtype: list of :class:`.Contig`
        """
        return self._contigs

    @property
    def x_contig_names(self):
        """X contig names

        :rtype: list of str
        """
        return self._x_contig_names

    @property
    def y_contig_names(self):
        """Y contig names

        :rtype: list of str
        """
        return self._y_contig_names

    @property
    def mt_contig_names(self):
        """Mitochondrial contig names

        :rtype: list of str
        """
        return self._mt_contig_names

    @property
    def par(self):
        """Pseudoautosomal regions

        :rtype: list of :class:`.Interval`
        """
        return self._par

    @staticmethod
    @handle_py4j
    def GRCh37():
        """Genome reference for GRCh37

        Data from `<ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37.dict>`_

        >>> gr37 = GenomeReference.GRCh37()

        :rtype: :class:`.GenomeReference`
        """
        return GenomeReference._from_java(Env.hail().variant.GenomeReference.GRCh37())

    @staticmethod
    @handle_py4j
    def GRCh38():
        """Genome reference for GRCh38

        Data from `<ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/Homo_sapiens_assembly38.dict>`_

        >>> gr38 = GenomeReference.GRCh38()

        :rtype: :class:`.GenomeReference`
        """
        return GenomeReference._from_java(Env.hail().variant.GenomeReference.GRCh38())

    @handle_py4j
    def _init_from_java(self, jrep):
        self._jrep = jrep

    @classmethod
    def _from_java(cls, jrep):
        gr = GenomeReference.__new__(cls)
        gr._init_from_java(jrep)
        gr._name = jrep.name()
        gr._contigs = [Contig._from_java(x) for x in jrep.contigs()]
        gr._x_contig_names = [str(x) for x in jiterable_to_list(jrep.xContigs())]
        gr._y_contig_names = [str(x) for x in jiterable_to_list(jrep.yContigs())]
        gr._mt_contig_names = [str(x) for x in jiterable_to_list(jrep.mtContigs())]
        gr._par = [Interval._from_java(x) for x in jrep.par()]
        return gr
