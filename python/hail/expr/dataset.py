from __future__ import print_function  # Python 2 and 3 print compatibility

import warnings

from decorator import decorator
from hail.expr.column import *


warnings.filterwarnings(module=__name__, action='once')


class DatasetTemplate(object):
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

        self.__setattr__("g", convert_column(Column("g", self.genotype_schema)))
        self.__setattr__("v", convert_column(Column("v", self.rowkey_schema)))
        self.__setattr__("s", convert_column(Column("s", self.colkey_schema)))
        self.__setattr__("va", convert_column(Column("va", self.variant_schema)))
        self.__setattr__("sa", convert_column(Column("sa", self.sample_schema)))
        self.__setattr__("globals", convert_column(Column("global", self.global_schema)))
        self.__setattr__("gs", convert_column(Column("gs", TAggregable(self.genotype_schema))))

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @property
    def _jvdf(self):
        if self._jvdf_cache is None:
            self._jvdf_cache = Env.hail().variant.VariantDatasetFunctions(self._jvds.toVDS())
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
    def __init__(self, hc, jvds):
        super(AggregatedVariantDataset, self).__init__(hc, jvds)
        self.__setattr__("variants", convert_column(Column("variants", TAggregable(self.variant_schema))))
        self.__setattr__("samples", convert_column(Column("samples", TAggregable(self.sample_schema))))

    @handle_py4j
    def query_samples_typed(self, exprs):
        if isinstance(exprs, list):
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jvds.querySamples(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes
        else:
            result = self._jvds.querySamples(to_expr(exprs))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    def query_samples(self, exprs):
        r, t = self.query_samples_typed(exprs)
        return r

    @handle_py4j
    def query_variants_typed(self, exprs):
        if isinstance(exprs, list):
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jvds.queryVariants(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes

        else:
            result = self._jvds.queryVariants(to_expr(exprs))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    def query_variants(self, exprs):
        r, t = self.query_variants_typed(exprs)
        return r

    @handle_py4j
    def query_genotypes_typed(self, exprs):
        if isinstance(exprs, list):
            exprs = [e.expr for e in exprs]
            result_list = self._jvds.queryGenotypes(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes
        else:
            result = self._jvds.queryGenotypes(to_expr(exprs))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    def query_genotypes(self, exprs):
        r, t = self.query_genotypes_typed(exprs)
        return r


class NewVariantDataset(DatasetTemplate):

    @handle_py4j
    def count_variants(self):
        return self._jvds.countVariants()

    @handle_py4j
    def update_genotypes(self, f):
        exprs = ", ".join(["g = " + to_expr(f(self.g))])
        jvds = self._jvds.annotateGenotypesExpr(exprs)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def with_alleles(self, propagate_gq=False, **kwargs):
        exprs = ", ".join(["va." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvdf.annotateAllelesExpr(exprs, propagate_gq)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def with_genotypes(self, **kwargs):
        exprs = ", ".join(["g." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateGenotypesExpr(exprs)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def with_global(self, **kwargs):
        exprs = ", ".join(["global." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateGlobalExpr(exprs)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def with_samples(self, **kwargs):
        exprs = ", ".join(["sa." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateSamplesExpr(exprs)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def with_variants(self, **kwargs):
        exprs = ", ".join(["va." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvds.annotateVariantsExpr(exprs)
        return NewVariantDataset(self.hc, jvds)

    # FIXME: Filter alleles will be rewritten so as not to include an annotation path

    @handle_py4j
    def filter_genotypes(self, expr, keep=True):
        jvds = self._jvds.filterGenotypes(expr.expr, keep)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def filter_samples(self, expr, keep=True):
        jvds = self._jvds.filterSamplesExpr(expr.expr, keep)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def filter_variants(self, expr, keep=True):
        jvds = self._jvds.filterVariantsExpr(expr.expr, keep)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def aggregate(self):
        return AggregatedVariantDataset(self.hc, self._jvds)

    def _to_old_variant_dataset(self):
        from hail.dataset import VariantDataset
        return VariantDataset(self.hc, self._jvds)
