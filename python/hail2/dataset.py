from __future__ import print_function  # Python 2 and 3 print compatibility

from hail2.expr.column import *
from hail.history import *
from hail.typecheck import *


class DatasetTemplate(HistoryMixin):
    def __init__(self, hc, jvds):
        self.hc = hc
        self._jvds = jvds

        self._globals = None
        self._sample_annotations = None
        self._colkey_schema = None
        self._sa_schema = None
        self._rowkey_schema = None
        self._va_schema = None
        self._global_schema = None
        self._genotype_schema = None
        self._sample_ids = None
        self._num_samples = None
        self._jvdf_cache = None
        self._scope = Scope()

        attrs = [("g", "g", self.genotype_schema),
                 ("v", "v", self.rowkey_schema),
                 ("s", "s", self.colkey_schema),
                 ("va", "va", self.variant_schema),
                 ("sa", "sa", self.sample_schema),
                 ("globals", "global", self.global_schema)]

        for n_python, n_scala, t in attrs:
            column = convert_column(Column(n_scala, t))
            self.__setattr__(n_python, column)
            self._scope.__setattr__(n_python, column)

        self.__setattr__("gs", convert_column(Column("gs", TAggregable(self.genotype_schema), scope=self._scope)))

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @property
    def _jvdf(self):
        if self._jvdf_cache is None:
            self._jvdf_cache = Env.hail().variant.VariantDatasetFunctions(self._jvds)
        return self._jvdf_cache

    @property
    def global_schema(self):
        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jvds.globalSignature())
        return self._global_schema

    @property
    def colkey_schema(self):
        if self._colkey_schema is None:
            self._colkey_schema = Type._from_java(self._jvds.sSignature())
        return self._colkey_schema

    @property
    def sample_schema(self):
        if self._sa_schema is None:
            self._sa_schema = Type._from_java(self._jvds.saSignature())
        return self._sa_schema

    @property
    def rowkey_schema(self):
        if self._rowkey_schema is None:
            self._rowkey_schema = Type._from_java(self._jvds.vSignature())
        return self._rowkey_schema

    @property
    def variant_schema(self):
        if self._va_schema is None:
            self._va_schema = Type._from_java(self._jvds.vaSignature())
        return self._va_schema

    @property
    def genotype_schema(self):
        if self._genotype_schema is None:
            self._genotype_schema = Type._from_java(self._jvds.genotypeSignature())
        return self._genotype_schema


class AggregatedVariantDataset(DatasetTemplate):
    def __init__(self, hc, jvds, scope):
        super(AggregatedVariantDataset, self).__init__(hc, jvds)
        self.__setattr__("variants", convert_column(Column("variants", TAggregable(self.variant_schema), scope=scope)))
        self.__setattr__("samples", convert_column(Column("samples", TAggregable(self.sample_schema), scope=scope)))

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_samples_typed(self, *exprs):
        if len(exprs) > 1:
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jvds.querySamples(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes
        else:
            result = self._jvds.querySamples(to_expr(exprs[0]))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_samples(self, *exprs):
        r, t = self.query_samples_typed(*exprs)
        return r

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_variants_typed(self, *exprs):
        if len(exprs) > 1:
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jvds.queryVariants(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes

        else:
            result = self._jvds.queryVariants(to_expr(exprs[0]))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_variants(self, *exprs):
        r, t = self.query_variants_typed(*exprs)
        return r

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_genotypes_typed(self, *exprs):
        if len(exprs) > 1:
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jvds.queryGenotypes(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes
        else:
            result = self._jvds.queryGenotypes(to_expr(exprs[0]))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=tupleof(Column))
    def query_genotypes(self, *exprs):
        r, t = self.query_genotypes_typed(*exprs)
        return r


class VariantDataset(DatasetTemplate):
    """Hail's primary representation of genomic data, a matrix keyed by sample and variant.

    .. testsetup::

        hc.stop()

    **Examples**

    Import vds:

    >>> from hail2 import *
    >>> hc = HailContext()
    >>> vds = hc.read("data/example.vds")

    Add annotations:

    >>> vds_ann = vds.annotate_global(foo = 5)

    >>> vds_ann = (vds_ann.annotate_variants(x1 = vds_ann.gs.count(),
    ...                                      x2 = vds_ann.gs.fraction(lambda g, _: False),
    ...                                      x3 = vds_ann.gs.filter(lambda g, _: True).count(),
    ...                                      x4 = vds_ann.va.info.AC + vds_ann.globals.foo)
    ...               .annotate_alleles(propagate_gq=False, a1 = vds_ann.gs.count()))

    >>> vds_ann = vds_ann.annotate_samples(apple = 6)
    >>> vds_ann = vds_ann.annotate_samples(x1 = vds_ann.gs.count(),
    ...                                    x2 = vds_ann.gs.fraction(lambda g, _: False),
    ...                                    x3 = vds_ann.gs.filter(lambda g, _: True).count(),
    ...                                    x4 = vds_ann.globals.foo + vds_ann.sa.apple) # Note `apple` annotation was created in separate function call

    >>> vds_ann = vds_ann.annotate_genotypes(x1 = vds_ann.va.x1 + vds_ann.globals.foo,
    ...                                      x2 = vds_ann.va.x1 + vds_ann.sa.x1 + vds_ann.globals.foo)


    Filter:

    >>> vds_filtered = (vds_ann.filter_variants((vds_ann.va.x1 == 5) & (vds_ann.gs.count() == 3) & (vds_ann.globals.foo == 2))
    ...                        .filter_samples((vds_ann.sa.x1 == 5) & (vds_ann.gs.count() == 3) & (vds_ann.globals.foo == 2), keep=False)
    ...                        .filter_genotypes((vds_ann.va.x1 == 5) & (vds_ann.sa.x1 == 5) & (vds_ann.globals.foo == 2) & (vds_ann.g.x1 != 3)))

    Update genotypes:

    >>> vds_updated = vds.update_genotypes(lambda g: Struct({'dp': g.dp, 'gq': g.gq}))

    Query:

    >>> import hail2.expr.functions as hf

    >>> vds_agg = vds_ann.aggregate()

    >>> qv = vds_agg.query_variants(vds_agg.variants.map(lambda v, _: v).count())
    >>> qs = vds_agg.query_samples(vds_agg.samples.map(lambda s, _: s).count())
    >>> qg = vds_agg.query_genotypes(vds_agg.gs.map(lambda g, _: g).count())

    >>> [qv1, qv2] = vds_agg.query_variants(vds_agg.variants.map(lambda v, _: _.v.contig).collect(),
    ...                                     vds_agg.variants.map(lambda v, _: _.va.x1).collect())

    >>> [qs1, qs2] = vds_agg.query_samples(vds_agg.samples.map(lambda s, _: s).collect(),
    ...                                    vds_agg.samples.map(lambda s, _: _.sa.x1).collect())

    >>> [qg1, qg2] = vds_agg.query_genotypes(vds_agg.gs.filter(lambda g, _: False).map(lambda g, _: _.sa.x1).collect(),
    ...                                      vds_agg.gs.filter(lambda g, _: hf.pcoin(0.1)).map(lambda g, _: g).collect())

    Convert to :class:`hail.VariantDataset`:

    >>> vds_h1 = vds.to_hail1()

    Run methods only available in :class:`hailVariantDataset`:

    >>> vds_h1 = vds.to_hail1()
    >>> vds_h1.sample_qc().samples_table().write("output/h1_sqc.kt")

    """

    @handle_py4j
    def count_variants(self):
        return self._jvds.countVariants()

    @handle_py4j
    @record_method
    def update_genotypes(self, f):
        exprs = ", ".join(["g = " + to_expr(f(self.g))])
        jvds = self._jvds.annotateGenotypesExpr(exprs)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(propagate_gq=bool,
                      kwargs=dictof(strlike, anytype))
    def annotate_alleles(self, propagate_gq=False, **kwargs):
        exprs = ", ".join(["va." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvdf.annotateAllelesExpr(exprs, propagate_gq)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(kwargs=dictof(strlike, anytype))
    def annotate_genotypes(self, **kwargs):
        exprs = ", ".join(["g." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateGenotypesExpr(exprs)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(kwargs=dictof(strlike, anytype))
    def annotate_global(self, **kwargs):
        exprs = ", ".join(["global." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateGlobalExpr(exprs)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(kwargs=dictof(strlike, anytype))
    def annotate_samples(self, **kwargs):
        exprs = ", ".join(["sa." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateSamplesExpr(exprs)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(kwargs=dictof(strlike, anytype))
    def annotate_variants(self, **kwargs):
        exprs = ", ".join(["va." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateVariantsExpr(exprs)
        return VariantDataset(self.hc, jvds)

    # FIXME: Filter alleles will be rewritten so as not to include an annotation path

    @handle_py4j
    @record_method
    @typecheck_method(expr=oneof(bool, BooleanColumn),
                      keep=bool)
    def filter_genotypes(self, expr, keep=True):
        jvds = self._jvds.filterGenotypes(to_expr(expr), keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(expr=oneof(bool, BooleanColumn),
                      keep=bool)
    def filter_samples(self, expr, keep=True):
        jvds = self._jvds.filterSamplesExpr(to_expr(expr), keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    @typecheck_method(expr=oneof(bool, BooleanColumn),
                      keep=bool)
    def filter_variants(self, expr, keep=True):
        jvds = self._jvds.filterVariantsExpr(to_expr(expr), keep)
        return VariantDataset(self.hc, jvds)

    @handle_py4j
    @record_method
    def aggregate(self):
        return AggregatedVariantDataset(self.hc, self._jvds, self._scope)

    @handle_py4j
    @write_history('output', is_dir=True)
    @typecheck_method(output=strlike,
                      overwrite=bool)
    def write(self, output, overwrite=False):
        self._jvds.write(output, overwrite)

    @record_method
    def to_hail1(self):
        import hail
        return hail.VariantDataset(self.hc, self._jvds)
