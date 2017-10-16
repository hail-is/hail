from __future__ import print_function  # Python 2 and 3 print compatibility

from hail2.expr.column import *
from hail.java import *
from hail.typ import Type, TArray, TStruct, TAggregable
from hail.representation import Struct
from hail.typecheck import *
from hail.utils import wrap_to_list
from pyspark.sql import DataFrame
from hail.history import *
from hail.typecheck import *


class KeyTableTemplate(HistoryMixin):
    def __init__(self, hc, jkt):
        self.hc = hc
        self._jkt = jkt

        self._schema = None
        self._num_columns = None
        self._key = None
        self._column_names = None

    def __getitem__(self, item):
        if item in self.columns:
            return self.columns[item]
        else:
            raise "Could not find column `" + item + "' in key table."

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, item):
        pass

    def __repr__(self):
        return self._jkt.toString()

    @property
    @handle_py4j
    def schema(self):
        if self._schema is None:
            self._schema = Type._from_java(self._jkt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema


class GroupedKeyTable(KeyTableTemplate):
    """KeyTable that has been grouped.
    """

    def __init__(self, hc, jkt, groups, scope):
        super(GroupedKeyTable, self).__init__(hc, jkt)
        self._groups = groups

        for fd in self.schema.fields:
            self.__setattr__(fd.name, convert_column(Column(fd.name, TAggregable(fd.typ), scope=scope)))

    @property
    def groups(self):
        return self._groups

    @handle_py4j
    @record_method
    @typecheck_method(num_partitions=nullable(integral),
                      kwargs=dictof(strlike, anytype))
    def aggregate_by_key(self, num_partitions=None, **kwargs):
        """Aggregate columns programmatically by key.

        :param num_partitions: Target number of partitions in the resulting table.
        :type num_partitions: int or None

        :param kwargs: Annotation expression with the left hand side equal to the new column name and the right hand side is any type.
        :type kwargs: dict of str to anytype

        :return: Key table with new columns specified by ``kwargs`` that have been aggregated by the key specified by groups.
        :rtype: :class:`.KeyTable`
        """
        agg_expr = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return KeyTable(self.hc, self._jkt.aggregate(self._groups, ", ".join(agg_expr), joption(num_partitions)))


class AggregatedKeyTable(KeyTableTemplate):
    """KeyTable that has been aggregated.

    .. testsetup::

        hc.stop()
        import hail2 as h2
        hc = h2.HailContext()
        kt1 = hc.import_table('data/kt_example1.tsv', impute=True)
    """

    def __init__(self, hc, jkt, scope):
        super(AggregatedKeyTable, self).__init__(hc, jkt)
        for fd in self.schema.fields:
            self.__setattr__(fd.name, convert_column(Column(fd.name, TAggregable(fd.typ), scope=scope)))

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_typed(self, *exprs):
        """Performs aggregation queries over columns of the table, and returns Python object(s) and types.

        **Examples**

        >>> kt1_agg = kt1.aggregate()

        >>> mean_value, t = kt1_agg.query_typed(kt1_agg.C1.stats().mean)

        >>> [hist, counter], [t1, t2] = kt1_agg.query_typed(kt1_agg.HT.hist(50, 80, 10),
        ...                                                 kt1_agg.SEX.counter())

        See :py:meth:`~.AggregatedKeyTable.query` for more information.

        :param exprs: Columns to query. Multiple arguments allowed.
        :type exprs: :class:`.Column`

        :rtype: (annotation or list of annotation,  :class:`.Type` or list of :class:`.Type`)
        """

        if len(exprs) > 1:
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jkt.query(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes

        else:
            result = self._jkt.query(to_expr(exprs[0]))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query(self, *exprs):
        """Performs aggregation queries over columns of the table, and returns Python object(s).

        **Examples**

        Convert table to aggregated form:

        >>> kt1_agg = kt1.aggregate()

        Compute the mean value of C1:

        >>> mean_value = kt1_agg.query(kt1_agg.C1.stats().mean)

        Compute a histogram for HT and a counter for SEX:

        >>> [hist, counter] = kt1_agg.query(kt1_agg.HT.hist(50, 80, 10),
        ...                                 kt1_agg.SEX.counter())

        Compute the fraction of males with a height greater than 70:

        >>> fraction_tall_male = (kt1_agg.query(kt1_agg.HT.filter(lambda x, _: _.SEX == "M")
        ...                                               .fraction(lambda x, _: x > 70)))

        Return a list of ids where C2 < C3:

        >>> ids = kt1_agg.query(kt1_agg.ID.filter(lambda x, _: _.C2 < _.C3).collect())

        :param exprs: Columns to query. Multiple arguments allowed.
        :type exprs: :class:`.Column`

        :rtype: annotation or list of annotation
        """

        r, t = self.query_typed(*exprs)
        return r


class KeyTable(KeyTableTemplate):
    """Hail's version of a SQL table where columns can be designated as keys.

    Key tables may be imported from a text file or Spark DataFrame with :py:meth:`~hail2.HailContext.import_table`
    or :py:meth:`~hail2.KeyTable.from_dataframe`, generated from a variant dataset
    with :py:meth:`~hail2.VariantDataset.make_table`, :py:meth:`~hail2.VariantDataset.genotypes_table`,
    :py:meth:`~hail2.VariantDataset.samples_table`, or :py:meth:`~hail2.VariantDataset.variants_table`.

    In the examples below, we have imported two key tables from text files (``kt1`` and ``kt2``).

    .. testsetup::

        hc.stop()
        import hail2 as h2
        from hail2 import *
        hc = h2.HailContext()

    >>> kt1 = hc.import_table('data/kt_example1.tsv', impute=True)

    +--+---+---+-+-+----+----+----+
    |ID|HT |SEX|X|Z| C1 | C2 | C3 |
    +==+===+===+=+=+====+====+====+
    |1 |65 |M  |5|4|2	|50  |5   |
    +--+---+---+-+-+----+----+----+
    |2 |72 |M  |6|3|2	|61  |1   |
    +--+---+---+-+-+----+----+----+
    |3 |70 |F  |7|3|10	|81  |-5  |
    +--+---+---+-+-+----+----+----+
    |4 |60 |F  |8|2|11	|90  |-10 |
    +--+---+---+-+-+----+----+----+

    >>> kt2 = hc.import_table('data/kt_example2.tsv', impute=True)

    +---+---+------+
    |ID	|A  |B     |
    +===+===+======+
    |1	|65 |cat   |
    +---+---+------+
    |2	|72 |dog   |
    +---+---+------+
    |3	|70 |mouse |
    +---+---+------+
    |4	|60 |rabbit|
    +---+---+------+

    **Examples**

    >>> schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    ...                  [TInt32(), TInt32(), TInt32(), TInt32(), TString(),
    ...                   TArray(TInt32()), TArray(TStruct(['x', 'y', 'z'], [TInt32(), TInt32(), TString()])),
    ...                   TStruct(['a', 'b', 'c'], [TInt32(), TInt32(), TString()]),
    ...                   TBoolean(), TStruct(['x', 'y', 'z'], [TInt32(), TInt32(), TString()])])

    >>> rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5,
    ...          'e': "hello", 'f': [1, 2, 3],
    ...          'g': [Struct({'x': 1, 'y': 5, 'z': "banana"})],
    ...          'h': Struct({'a': 5, 'b': 3, 'c': "winter"}),
    ...          'i': True,
    ...          'j': Struct({'x': 3, 'y': 2, 'z': "summer"})}]

    >>> kt = KeyTable.parallelize(rows, schema)

    Select columns:

    >>> kt.select(True, kt.a, kt.e, *kt.h, foo = kt.a + kt.b)

    Filter:

    >>> kt.filter(kt.a == 4)
    >>> kt.filter((kt.d == -1) | (kt.c == 20) | (kt.e == "hello"))
    >>> kt.filter((kt.c != 20) & (kt.a == 4))
    >>> kt.filter(True)

    Query rows:

    >>> kt_agg = kt.aggregate() # Must create an aggregated table first before querying
    >>> q1, q2 = kt_agg.query(kt_agg.b.sum(), kt_agg.b.count())
    >>> q3 = kt_agg.query(kt_agg.e.collect())
    >>> q4 = kt_agg.query(kt_agg.e.filter(lambda x, _: (_.d >= 5) | (_.a == 0)).collect())

    Annotate:

    >>> kt.annotate(foo = kt.a + 1, foo2 = kt.a)

    >>> import hail2.expr.functions as hf
    >>> kt.annotate(
    ...         x1 = kt.f.map(lambda x: x * 2),
    ...         x2 = kt.f.map(lambda x: [x, x + 1]).flat_map(lambda x: x),
    ...         x3 = kt.f.min(),
    ...         x4 = kt.f.max(),
    ...         x5 = kt.f.sum(),
    ...         x6 = kt.f.product(),
    ...         x7 = kt.f.length(),
    ...         x8 = kt.f.filter(lambda x: x == 3),
    ...         x9 = kt.f.tail(),
    ...         x10 = kt.f[:],
    ...         x11 = kt.f[1:2],
    ...         x12 = kt.f.map(lambda x: [x, x + 1]),
    ...         x13 = kt.f.map(lambda x: [[x, x + 1], [x + 2]]).flat_map(lambda x: x),
    ...         x14 = hf.where(kt.a < kt.b, kt.c, Column.null(TInt32())), # note setting else statement to NA with type Int32
    ...         x15 = set([1, 2, 3])
    ...     )

    >>> kt.annotate(
    ...         x1 = kt.a + 5,
    ...         x2 = 5 + kt.a,
    ...         x3 = kt.a + kt.b,
    ...         x4 = kt.a - 5,
    ...         x5 = 5 - kt.a,
    ...         x6 = kt.a - kt.b,
    ...         x7 = kt.a * 5,
    ...         x8 = 5 * kt.a,
    ...         x9 = kt.a * kt.b,
    ...         x10 = kt.a / 5,
    ...         x11 = 5 / kt.a,
    ...         x12 = kt.a / kt.b,
    ...         x13 = -kt.a,
    ...         x14 = +kt.a,
    ...         x15 = kt.a == kt.b,
    ...         x16 = kt.a == 5,
    ...         x17 = 5 == kt.a,
    ...         x18 = kt.a != kt.b,
    ...         x19 = kt.a != 5,
    ...         x20 = 5 != kt.a,
    ...         x21 = kt.a > kt.b,
    ...         x22 = kt.a > 5,
    ...         x23 = 5 > kt.a,
    ...         x24 = kt.a >= kt.b,
    ...         x25 = kt.a >= 5,
    ...         x26 = 5 >= kt.a,
    ...         x27 = kt.a < kt.b,
    ...         x28 = kt.a < 5,
    ...         x29 = 5 < kt.a,
    ...         x30 = kt.a <= kt.b,
    ...         x31 = kt.a <= 5,
    ...         x32 = 5 <= kt.a,
    ...         x33 = (kt.a == 0) & (kt.b == 5),
    ...         x34 = (kt.a == 0) | (kt.b == 5),
    ...         x35 = False,
    ...         x36 = True
    ...     )

    Annotate with functions: # FIXME: add link

    >>> import hail2.expr.functions as hf
    >>> kt.annotate(
    ...         chisq = hf.chisq(kt.a, kt.b, kt.c, kt.d),
    ...         combvar = hf.combine_variants(Variant.parse("1:2:A:T"), Variant.parse("1:2:A:C")),
    ...         ctt = hf.ctt(kt.a, kt.b, kt.c, kt.d, 5),
    ...         Dict = hf.Dict([kt.a, kt.b], [kt.c, kt.d]),
    ...         dpois = hf.dpois(4, kt.a),
    ...         drop = hf.drop(kt.h, 'b', 'c'),
    ...         exp = hf.exp(kt.c),
    ...         fet = hf.fet(kt.a, kt.b, kt.c, kt.d),
    ...         gt_index = hf.gt_index(kt.a, kt.b),
    ...         gtj = hf.gtj(kt.a),
    ...         gtk = hf.gtk(kt.b),
    ...         hwe = hf.hwe(1, 2, 1),
    ...         index = hf.index(kt.g, 'z'),
    ...         is_defined = hf.is_defined(kt.i),
    ...         is_missing = hf.is_missing(kt.i),
    ...         is_nan = hf.is_nan(kt.a.to_float64()),
    ...         json = hf.json(kt.g),
    ...         log = hf.log(kt.a.to_float64(), kt.b.to_float64()),
    ...         log10 = hf.log10(kt.c.to_float64()),
    ...         merge = hf.merge(kt.h, kt.j),
    ...         or_else = hf.or_else(kt.a, 5),
    ...         or_missing = hf.or_missing(kt.i, kt.j),
    ...         pchisqtail = hf.pchisqtail(kt.a.to_float64(), kt.b.to_float64()),
    ...         pcoin = hf.pcoin(0.5),
    ...         pnorm = hf.pnorm(0.2),
    ...         pow = hf.pow(2.0, kt.b),
    ...         ppois = hf.ppois(kt.a.to_float64(), kt.b.to_float64()),
    ...         qchisqtail = hf.qchisqtail(kt.a.to_float64(), kt.b.to_float64()),
    ...         range = hf.range(0, 5, kt.b),
    ...         rnorm = hf.rnorm(0.0, kt.b),
    ...         rpois = hf.rpois(kt.a),
    ...         runif = hf.runif(kt.b, kt.a),
    ...         select = hf.select(kt.h, 'c', 'b'),
    ...         sqrt = hf.sqrt(kt.a),
    ...         to_str = [hf.to_str(5), hf.to_str(kt.a), hf.to_str(kt.g)],
    ...         where = hf.where(kt.i, 5, 10) # equivalent of if-else
    ...     )

    Construct Genotype, Call, Variant, Locus, and Intervals:

    >>> from hail2.expr.column import *

    >>> schema = TStruct(['a', 'b', 'c', 'd'], [TFloat64(), TFloat64(), TInt32(), TInt64()])
    >>> rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
    >>> ktx = KeyTable.parallelize(rows, schema)

    >>> ktx = ktx.annotate(v1 = VariantColumn.parse("1:500:A:T"),
    ...                    v2 = VariantColumn.from_args("1", 23, "A", "T"),
    ...                    v3 = VariantColumn.from_args("1", 23, "A", ["T", "G"]),
    ...                    l1 = LocusColumn.parse("1:51"),
    ...                    l2 = LocusColumn.from_args("1", 51),
    ...                    i1 = IntervalColumn.parse("1:51-56"),
    ...                    i2 = IntervalColumn.from_args("1", 51, 56),
    ...                    i3 = IntervalColumn.from_loci(LocusColumn.from_args("1", 51), LocusColumn.from_args("1", 56)))

    >>> ktx = ktx.annotate(g1 = GenotypeColumn.dosage_genotype(ktx.v1, [0.0, 1.0, 0.0]),
    ...                    g2 = GenotypeColumn.dosage_genotype(ktx.v1, [0.0, 1.0, 0.0], call=CallColumn.from_int32(1)),
    ...                    g3 = GenotypeColumn.from_call(CallColumn.from_int32(1)),
    ...                    g4 = GenotypeColumn.pl_genotype(ktx.v1, CallColumn.from_int32(1), [6, 7], 13, 20, [20, 0, 1000]))


    Aggregate by key:

    >>> schema = TStruct(['status', 'gt', 'qPheno'],
    ...                  [TInt32(), TGenotype(), TInt32()])

    >>> rows = [{'status':0, 'gt': Genotype(0), 'qPheno': 3},
    ...         {'status':0, 'gt': Genotype(1), 'qPheno': 13},
    ...         {'status':1, 'gt': Genotype(1), 'qPheno': 20}]

    >>> kty = KeyTable.parallelize(rows, schema)

    >>> kt_grp = kty.group_by(status = kty.status) # Must create a grouped table first

    >>> kt_grp.aggregate_by_key(
    ...     x1 = kt_grp.qPheno.map(lambda x, _: x * 2).collect(),
    ...     x2 = kt_grp.qPheno.flat_map(lambda x, _: [x, x + 1]).collect(),
    ...     x3 = kt_grp.qPheno.min(),
    ...     x4 = kt_grp.qPheno.max(),
    ...     x5 = kt_grp.qPheno.sum(),
    ...     x6 = kt_grp.qPheno.map(lambda x, _: x.to_int64()).product(),
    ...     x7 = kt_grp.qPheno.count(),
    ...     x8 = kt_grp.qPheno.filter(lambda x, _: x == 3).count(),
    ...     x9 = kt_grp.qPheno.fraction(lambda x, _: x == 1),
    ...     x10 = kt_grp.qPheno.map(lambda x, _: x.to_float64()).stats(),
    ...     x11 = kt_grp.gt.hardy_weinberg(),
    ...     x12 = kt_grp.gt.map(lambda x, _: x.gp).info_score(),
    ...     x13 = kt_grp.gt.inbreeding(lambda x, _: 0.1),
    ...     x14 = kt_grp.gt.call_stats(lambda g, _: Variant("1", 10000, "A", "T")),
    ...     x15 = kt_grp.gt.map(lambda g, _: Struct({'a': 5, 'b': "foo", 'c': Struct({'banana': 'apple'})})).collect()[0],
    ...     x16 = (kt_grp.gt.map(lambda g, _: Struct({'a': 5, 'b': "foo", 'c': Struct({'banana': 'apple'})}))
    ...            .map(lambda s, _: s.c.banana).collect()[0]),
    ...     num_partitions=5
    ... )

    Convert to :class:`hail.KeyTable`:

    >>> kt_h1 = kt.to_hail1()

    :ivar hc: Hail Context
    :vartype hc: :class:`~hail2.HailContext`
    """

    def __init__(self, hc, jkt):
        super(KeyTable, self).__init__(hc, jkt)

        self._scope = Scope()

        for fd in self.schema.fields:
            column = convert_column(Column(fd.name, fd.typ))
            self.__setattr__(fd.name, column)
            self._scope.__setattr__(fd.name, column)

    @property
    @handle_py4j
    def columns(self):
        if self._column_names is None:
            self._column_names = list(self._jkt.columns())
        return self._column_names

    @property
    @handle_py4j
    def num_columns(self):
        if self._num_columns is None:
            self._num_columns = self._jkt.nColumns()
        return self._num_columns

    @property
    @handle_py4j
    def key(self):
        if self._key is None:
            self._key = list(self._jkt.key())
        return self._key

    @handle_py4j
    def count(self):
        return self._jkt.count()

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(rows=oneof(listof(Struct), listof(dictof(strlike, anytype))),
                      schema=TStruct,
                      key=oneof(strlike, listof(strlike)),
                      num_partitions=nullable(integral))
    def parallelize(cls, rows, schema, key=[], num_partitions=None):
        return KeyTable(
            Env.hc(),
            Env.hail().keytable.KeyTable.parallelize(
                Env.hc()._jhc, [schema._convert_to_j(r) for r in rows],
                schema._jtype, wrap_to_list(key), joption(num_partitions)))

    @handle_py4j
    @record_method
    @typecheck_method(keys=tupleof(strlike))
    def key_by(self, *keys):
        """Change which columns are keys.

        **Examples**

        Assume ``kt`` is a :py:class:`.KeyTable` with three columns: c1, c2 and
        c3 and key c1.

        Change key columns:

        >>> kt_result = kt1.key_by('C2', 'C3')

        >>> kt_result = kt1.key_by('C2')

        Set to no keys:

        >>> kt_result = kt1.key_by()

        :param key: List of columns to be used as keys.
        :type key: str or list of str

        :return: Key table whose key columns are given by ``key``.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.keyBy(list(keys)))

    @handle_py4j
    @record_method
    @typecheck_method(kwargs=dictof(strlike, anytype))
    def annotate(self, **kwargs):
        """Add new columns.

        **Examples**

        Add new column ``Y`` which is equal to 5 times ``X``:

        >>> kt_result = kt1.annotate(Y = 5 * kt1.X)

        Add multiple columns simultaneously:

        >>> kt_result = kt1.annotate(A = kt1.X / 2,
        ...                          B = kt1.X + 21)

        Add new column ``A.B`` which is equal to ``X`` plus 3:

        >>> kt_result = kt1.annotate(**{'A.B': kt1.X + 3})


        **Notes**

        The scope for ``kwargs`` is all column names in the input :class:`KeyTable`.

        :param kwargs: Annotation expression with the left hand side equal to the new column name and the right hand side is any type.
        :type kwargs: dict of str to anytype

        :return: Key table with new columns specified by ``kwargs``.
        :rtype: :class:`.KeyTable`
        """

        exprs = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return KeyTable(self.hc, self._jkt.annotate(", ".join(exprs)))

    @handle_py4j
    @record_method
    @typecheck_method(expr=oneof(bool, BooleanColumn),
                      keep=bool)
    def filter(self, expr, keep=True):
        """Filter rows.

        **Examples**

        Keep rows where ``C1`` equals 5:

        >>> kt_result = kt1.filter(kt1.C1 == 5)

        Remove rows where ``C1`` equals 10:

        >>> kt_result = kt1.filter(kt1.C1 == 10, keep=False)

        **Notes**

        The scope for ``expr`` is all column names in the input :class:`KeyTable`.

        .. caution::
           When ``expr`` evaluates to missing, the row will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param expr: Boolean filter expression.
        :type expr: :class:`~hail2.expr.column.BooleanColumn` or bool

        :param bool keep: Keep rows where ``expr`` is true.

        :return: Filtered key table.
        :rtype: :class:`.KeyTable`
        """

        jkt = self._jkt.filter(to_expr(expr), keep)
        return KeyTable(self.hc, jkt)

    @handle_py4j
    @record_method
    @typecheck_method(qualified_name=bool,
                      exprs=tupleof(Column),
                      named_exprs=dictof(strlike, anytype))
    def select(self, qualified_name, *exprs, **named_exprs):
        """Select a subset of columns.

        **Examples**

        Assume ``kt1`` is a :py:class:`.KeyTable` with three columns: C1, C2 and
        C3.

        Select/drop columns:

        >>> kt_result = kt1.select(False, kt1.C1)

        Reorder the columns:

        >>> kt_result = kt1.select(False, kt1.C3, kt1.C1, kt1.C2)

        Drop all columns:

        >>> kt_result = kt1.select(False)

        Create a new column computed from existing columns:

        >>> kt_result = kt1.select(False, C_NEW = kt1.C1 + kt1.C2 + kt1.C3)

        :return: Key table with selected columns.
        :rtype: :class:`.KeyTable`
        """

        exprs = [to_expr(e) for e in exprs]
        exprs.extend([k + " = " + to_expr(v) for k, v in named_exprs.iteritems()])
        return KeyTable(self.hc, self._jkt.select(exprs, qualified_name))

    @handle_py4j
    @write_history('output', is_dir=False)
    def export(self, output, types_file=None, header=True):
        """Export to a TSV file.

        :param output:
        :param types_file:
        :param header:
        """
        self._jkt.export(output, types_file, header)

    @record_method
    def aggregate(self):
        """Convert column types to aggregables for querying.

        :return: Key table where columns are :class:`~hail2.expr.column.AggregableColumn`.
        :rtype: :class:`.AggregatedKeyTable`
        """

        return AggregatedKeyTable(self.hc, self._jkt, self._scope)

    @record_method
    @typecheck_method(kwargs=dictof(strlike, anytype))
    def group_by(self, **kwargs):
        """Group by key.

        :return: Key table where groupings are computed from expressions given by `kwargs`.
        :rtype: :class:`.GroupedKeyTable`
        """

        group_exprs = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return GroupedKeyTable(self.hc, self._jkt, ", ".join(group_exprs), self._scope)

    @handle_py4j
    @write_history('output', is_dir=True)
    @typecheck_method(output=strlike,
                      overwrite=bool)
    def write(self, output, overwrite=False):
        """Write as KT file.

        ***Examples***

        >>> kt1.write('output/kt1.kt')

        .. note:: The write path must end in ".kt".

        **Notes**

        A text file containing the python code to generate this output file is available at ``<output>/history.txt``.

        :param str output: Path of KT file to write.

        :param bool overwrite: If True, overwrite any existing KT file. Cannot be used
               to read from and write to the same path.
        """

        self._jkt.write(output, overwrite)

    @record_method
    def to_hail1(self):
        """Convert table to :class:`hail.KeyTable`.

        :rtype: :class:`hail.KeyTable`
        """
        import hail
        return hail.KeyTable(self.hc, self._jkt)
