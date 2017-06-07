from __future__ import print_function  # Python 2 and 3 print compatibility

import warnings

from decorator import decorator

from hail.types import Type, TGenotype, TString, TVariant
from hail.typecheck import *
from hail.java import *
from hail.expr.keytable_new import NewKeyTable
from hail.expr.column import *
from hail.keytable import KeyTable
from hail.representation import Interval, Pedigree, Variant
from hail.utils import Summary, wrap_to_list
from hail.kinshipMatrix import KinshipMatrix
from hail.ldMatrix import LDMatrix

# 1. Select/drop fields from va
# 2. order of kwargs is not preserved
# 3. drop columns
# 4. splat operator for export (this is handled by KeyTable operations)
# 5. cannot easily define default kwarg with **kwargs
# 6. with and select

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
            if self._is_generic_genotype:
                self._jvdf_cache = Env.hail().variant.GenericDatasetFunctions(self._jvds)
            else:
                self._jvdf_cache = Env.hail().variant.VariantDatasetFunctions(self._jvds)
        return self._jvdf_cache

    @property
    def _is_generic_genotype(self):
        return self._jvds.isGenericGenotype()

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


class GroupedVariantDataset(DatasetTemplate):
    def __init__(self, hc, jvds, groups):
        super(GroupedVariantDataset, self).__init__(hc, jvds)
        self._groups = groups

    @handle_py4j
    def aggregate_by_key(self, **kwargs):
        agg_exprs = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return NewKeyTable(self.hc, self._jvds.aggregateByKey(self._groups, agg_exprs))


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
    def update_genotypes(self, **kwargs):
        exprs = ", ".join(["g." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvdf.annotateGenotypesExpr(exprs)
        vds = NewVariantDataset(self.hc, jvds)
        if isinstance(vds.genotype_schema, TGenotype):
            return NewVariantDataset(self.hc, vds._jvdf.toVDS())
        else:
            return vds

    @handle_py4j
    def with_alleles(self, propagate_gq=False, **kwargs):
        exprs = ", ".join(["va." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvdf.annotateAllelesExpr(exprs, propagate_gq)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def with_genotypes(self, **kwargs):
        exprs = ", ".join(["g." + k + " = " + to_expr(v) for k, v in kwargs.items()])
        jvds = self._jvdf.annotateGenotypesExpr(exprs)
        vds = NewVariantDataset(self.hc, jvds)
        if isinstance(vds.genotype_schema, TGenotype):
            return NewVariantDataset(self.hc, vds._jvdf.toVDS())
        else:
            return vds

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

    # FIXME: This needs to be rethought
    # @handle_py4j
    # def filter_alleles(self, expr, annotation='va = va', subset=True, keep=True,
    #                    filter_altered_genotypes=False, max_shift=100, keep_star=False):
    #     jvds = self._jvdf.filterAlleles(expr.expr, annotation, filter_altered_genotypes, keep, subset, max_shift,
    #                                     keep_star)
    #     return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def filter_genotypes(self, expr, keep=True):
        jvds = self._jvdf.filterGenotypes(expr.expr, keep)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def filter_samples(self, expr, keep=True):
        jvds = self._jvds.filterSamplesExpr(expr.expr, keep)
        return NewVariantDataset(self.hc, jvds)

    @handle_py4j
    def filter_variants(self, expr, keep=True):
        jvds = self._jvds.filterVariantsExpr(expr.expr, keep)
        return NewVariantDataset(self.hc, jvds)

    def aggregate(self):
        return AggregatedVariantDataset(self.hc, self._jvds)

    def group_by(self, **kwargs):
        group_exprs = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return GroupedVariantDataset(self.hc, self._jvds, ", ".join(group_exprs))
