from __future__ import print_function  # Python 2 and 3 print compatibility

from hail2.expr.expression import *
from hail2.expr.ast import *
from hail.history import *
from hail.typecheck import *
import hail2


class GroupedMatrix(object):
    def __init__(self, parent, group, grouped_indices):
        self._parent = parent
        self._group = group
        self._grouped_indices = grouped_indices
        self._partitions = None
        self._fields = {}

        for f in parent._fields:
            self._set_field(f, parent._fields[f])

    def set_partitions(self, n):
        self._partitions = n
        return self

    def _set_field(self, key, value):
        assert key not in self._fields, key
        self._fields[key] = value
        if key in dir(self):
            warn("Name collision: field '{}' already in object dict."
                 " This field must be referenced with indexing syntax".format(key))
        else:
            self.__dict__[key] = value

    @handle_py4j
    def aggregate(self, **named_exprs):
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        strs = []

        base, cleanup = self._parent._process_joins(*((self._group,) + tuple(named_exprs.values())))
        for k, v in named_exprs.items():
            analyze(v, self._grouped_indices, {self._parent._row_axis, self._parent._col_axis},
                    set(self._parent._fields.keys()))
            replace_aggregables(v._ast, 'gs')

        struct_expr = to_expr(Struct(**named_exprs))
        group_str = self._group._ast.to_hql()

        if self._grouped_indices == self._parent._row_indices:
            # group variants
            return cleanup(
                Matrix(self._parent._hc, base._jvds.groupVariantsBy(group_str, struct_expr._ast.to_hql(), True)))
        else:
            assert self._grouped_indices == self._parent._col_indices
            # group samples
            raise NotImplementedError()


class Matrix(object):
    """Hail's representation of a structured matrix.

    **Examples**

    Read a matrix:

    >>> from hail2 import *
    >>> hc = HailContext()
    >>> m = hc.import_vcf('data/example2.vcf.bgz', generic=True)

    Add annotations:

    >>> m = m.annotate_globals(pli={'SCN1A': 0.999, 'SONIC': 0.014},
    ...                        populations = ['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])

    >>> m = m.annotate_cols(pop = m.populations[f.runif(0, 6).to_int32()],
    ...                     sample_gq = f.mean(m.GQ),
    ...                     sample_dp = f.mean(m.DP))

    >>> m = m.annotate_rows(variant_gq = f.mean(m.GQ),
    ...                     variant_dp = f.mean(m.GQ),
    ...                     sas_hets = f.count_where(m.GT.is_het()))

    >>> m = m.annotate_entries(gq_by_dp = m.GQ / m.DP)

    Filter:

    >>> m = m.filter_cols(m.pop != 'EUR')

    >>> m = m.filter_rows((m.variant_gq > 10) & (m.variant_dp > 5))

    >>> m = m.filter_entries(m.gq_by_dp > 1)

    Query:

    >>> col_stats = m.aggregate_cols(pop_counts = f.counter(m.pop),
    ...                              high_quality = f.fraction((m.sample_gq > 10) & (m.sample_dp > 5)))
    >>> print(col_stats.pop_counts)
    >>> print(col_stats.high_quality)

    >>> row_stats = m.aggregate_rows(het_dist = f.stats(m.sas_hets))
    >>> print(row_stats.het_dist)

    >>> entry_stats = m.aggregate_entries(call_rate = f.fraction(m.GT.is_called()),
    ...                                   global_gq_mean = f.mean(m.GQ))
    >>> print(entry_stats.call_rate)
    >>> print(entry_stats.global_gq_mean)
    """

    def __init__(self, hc, jvds):
        self._hc = hc
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
        self._row_axis = 'row'
        self._col_axis = 'column'
        self._global_indices = Indices(self, set())
        self._row_indices = Indices(self, {self._row_axis})
        self._col_indices = Indices(self, {self._col_axis})
        self._entry_indices = Indices(self, {self._row_axis, self._col_axis})
        self._reserved = {'v', 's'}
        self._fields = {}

        assert isinstance(self.global_schema, TStruct), self.col_schema
        assert isinstance(self.col_schema, TStruct), self.col_schema
        assert isinstance(self.row_schema, TStruct), self.row_schema
        assert isinstance(self.entry_schema, TStruct), self.entry_schema

        self._set_field('v', convert_expr(Expression(Reference('v'), self.rowkey_schema, self._row_indices)))
        self._set_field('s', convert_expr(Expression(Reference('s'), self.colkey_schema, self._col_indices)))

        for f in self.global_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name,
                            convert_expr(Expression(Select(Reference('global'), f.name), f.typ, self._global_indices)))

        for f in self.col_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, convert_expr(Expression(Select(Reference('sa'), f.name), f.typ,
                                                            self._col_indices)))

        for f in self.row_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, convert_expr(Expression(Select(Reference('va'), f.name), f.typ,
                                                            self._row_indices)))

        for f in self.entry_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, convert_expr(Expression(Select(Reference('g'), f.name), f.typ,
                                                            self._entry_indices)))

    def _set_field(self, key, value):
        assert key not in self._fields, key
        self._fields[key] = value
        if key in dir(self):
            warn("Name collision: field '{}' already in object dict."
                 " This field must be referenced with indexing syntax".format(key))
        else:
            self.__dict__[key] = value

    def __delattr__(self, item):
        if not item[0] == '_':
            raise NotImplementedError('Dataset objects are not mutable')

    def __setattr__(self, key, value):
        if not key[0] == '_':
            raise NotImplementedError('Dataset objects are not mutable')
        self.__dict__[key] = value

    @typecheck_method(item=oneof(strlike, sized_tupleof(oneof(slice, Expression), oneof(slice, Expression))))
    def __getitem__(self, item):
        if isinstance(item, str) or isinstance(item, unicode):
            if item in self._fields:
                return self._fields[item]
            else:
                raise KeyError("Dataset has no field '{}'".format(item))
        else:
            # this is the join path
            exprs = item
            row_key = None
            if isinstance(exprs[0], slice):
                s = exprs[0]
                if not (s.start is None and s.stop is None and s.step is None):
                    raise ExpressionException(
                        "Expect unbounded slice syntax ':' to indicate axes of a Matrix, but found parameter(s) [{}]".format(
                            ', '.join(x for x in ['start' if s.start is not None else None,
                                                  'stop' if s.stop is not None else None,
                                                  'step' if s.step is not None else None] if x is not None)
                        )
                    )
            else:
                row_key = to_expr(exprs[0])
                if row_key._type != self.rowkey_schema:
                    raise ExpressionException(
                        'Type mismatch for Matrix row key: expected key type {}, found {}'.format(
                            str(self.rowkey_schema), str(row_key._type)))

            col_key = None
            if isinstance(exprs[1], slice):
                s = exprs[1]
                if not (s.start is None and s.stop is None and s.step is None):
                    raise ExpressionException(
                        "Expect unbounded slice syntax ':' to indicate axes of a Matrix, but found parameter(s) [{}]".format(
                            ', '.join(x for x in ['start' if s.start is not None else None,
                                                  'stop' if s.stop is not None else None,
                                                  'step' if s.step is not None else None] if x is not None)
                        )
                    )
            else:
                col_key = to_expr(exprs[1])
                if col_key._type != self.colkey_schema:
                    raise ExpressionException(
                        'Type mismatch for Matrix col key: expected key type {}, found {}'.format(
                            str(self.colkey_schema), str(col_key._type)))

            if row_key is not None and col_key is not None:
                return self.index_entries(row_key, col_key)
            elif row_key is not None and col_key is None:
                return self.index_rows(row_key)
            elif row_key is None and col_key is not None:
                return self.index_cols(col_key)
            else:
                return self.index_globals()

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
    def col_schema(self):
        if self._sa_schema is None:
            self._sa_schema = Type._from_java(self._jvds.saSignature())
        return self._sa_schema

    @property
    def rowkey_schema(self):
        if self._rowkey_schema is None:
            self._rowkey_schema = Type._from_java(self._jvds.vSignature())
        return self._rowkey_schema

    @property
    def row_schema(self):
        if self._va_schema is None:
            self._va_schema = Type._from_java(self._jvds.vaSignature())
        return self._va_schema

    @property
    def entry_schema(self):
        if self._genotype_schema is None:
            self._genotype_schema = Type._from_java(self._jvds.genotypeSignature())
        return self._genotype_schema

    @handle_py4j
    def annotate_globals(self, **named_exprs):
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze(v, self._global_indices, set(), {'globals'})
            exprs.append('global.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
            self._check_field_name(k, self._global_indices)
        m = Matrix(self._hc, base._jvds.annotateGlobalExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def annotate_rows(self, **named_exprs):
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze(v, self._row_indices, {self._col_axis}, set(self._fields.keys()))
            replace_aggregables(v._ast, 'gs')
            exprs.append('va.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
            self._check_field_name(k, self._row_indices)
        m = Matrix(self._hc, base._jvds.annotateVariantsExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def annotate_cols(self, **named_exprs):
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze(v, self._col_indices, {self._row_axis}, set(self._fields.keys()))
            replace_aggregables(v._ast, 'gs')
            exprs.append('sa.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
            self._check_field_name(k, self._col_indices)
        m = Matrix(self._hc, base._jvds.annotateSamplesExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def annotate_entries(self, **named_exprs):
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze(v, self._entry_indices, set(), set(self._fields.keys()))
            exprs.append('g.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
            self._check_field_name(k, self._entry_indices)
        m = Matrix(self._hc, base._jvds.annotateGenotypesExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def select_globals(self, *exprs, **named_exprs):
        exprs = tuple(to_expr(e) for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))
        for e in exprs:
            all_exprs.append(e)
            analyze(e, self._global_indices, set(), {'globals'})
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append('`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name, e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze(e, self._global_indices, set(), {'globals'})
            self._check_field_name(k, self._global_indices)
            strs.append('`{}`: {}'.format(k, to_expr(e)._ast.to_hql()))
        m = Matrix(self._hc, base._jvds.annotateGlobalExpr('global = {' + ',\n'.join(strs) + '}'))
        return cleanup(m)

    @handle_py4j
    def select_cols(self, *exprs, **named_exprs):
        exprs = tuple(to_expr(e) for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze(e, self._col_indices, {self._row_axis}, set(self._fields.keys()))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_cols' expects keyword arguments for complex expressions")
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name,
                                          e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze(e, self._col_indices, {self._row_axis}, set(self._fields.keys()))
            self._check_field_name(k, self._col_indices)
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(k, e._ast.to_hql()))

        m = Matrix(self._hc, base._jvds.annotateSamplesExpr('sa = {' + ',\n'.join(strs) + '}'))
        return cleanup(m)

    @handle_py4j
    def select_rows(self, *exprs, **named_exprs):
        exprs = tuple(to_expr(e) for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze(e, self._row_indices, {self._col_axis}, set(self._fields.keys()))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_rows' expects keyword arguments for complex expressions")
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name,
                                          e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze(e, self._row_indices, {self._col_axis}, set(self._fields.keys()))
            self._check_field_name(k, self._row_indices)
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(k, e._ast.to_hql()))
        m = Matrix(self._hc, base._jvds.annotateVariantsExpr('va = {' + ',\n'.join(strs) + '}'))
        return cleanup(m)

    @handle_py4j
    def select_entries(self, *exprs, **named_exprs):
        exprs = tuple(to_expr(e) for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze(e, self._entry_indices, set(), set(self._fields.keys()))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append('`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name, e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze(e, self._entry_indices, set(), set(self._fields.keys()))
            self._check_field_name(k, self._entry_indices)
            strs.append('`{}`: {}'.format(k, e._ast.to_hql()))
        m = Matrix(self._hc, base._jvds.annotateGenotypesExpr('g = {' + ',\n'.join(strs) + '}'))
        return cleanup(m)

    @handle_py4j
    @typecheck_method(exprs=tupleof(oneof(strlike, Expression)))
    def drop(self, *exprs):
        """Drop fields from the matrix."""

        all_field_exprs = {e: k for k, e in self._fields.items()}
        fields_to_drop = set()
        for e in exprs:
            if isinstance(e, Expression):
                if e in all_field_exprs:
                    fields_to_drop.add(all_field_exprs[e])
                else:
                    raise ExpressionException("method 'drop' expects string field names or top-level field expressions"
                                              " (e.g. 'foo', matrix.foo, or matrix['foo'])")
            else:
                assert isinstance(e, str) or isinstance(str, unicode)
                if e not in self._fields:
                    raise IndexError("matrix has no field '{}'".format(e))
                fields_to_drop.add(e)

        m = self
        if any(self._fields[field]._indices == self._global_indices for field in fields_to_drop):
            # need to drop globals
            new_global_fields = {k.name: m._fields[k.name] for k in m.global_schema.fields if
                                 k.name not in fields_to_drop}
            m = m.select_globals(**new_global_fields)

        if any(self._fields[field]._indices == self._row_indices for field in fields_to_drop):
            # need to drop row fields
            new_row_fields = {k.name: m._fields[k.name] for k in m.row_schema.fields if k.name not in fields_to_drop}
            m = m.select_rows(**new_row_fields)

        if any(self._fields[field]._indices == self._col_indices for field in fields_to_drop):
            # need to drop col fields
            new_col_fields = {k.name: m._fields[k.name] for k in m.col_schema.fields if k.name not in fields_to_drop}
            m = m.select_cols(**new_col_fields)

        if any(self._fields[field]._indices == self._entry_indices for field in fields_to_drop):
            # need to drop entry fields
            new_entry_fields = {k.name: m._fields[k.name] for k in m.entry_schema.fields if
                                k.name not in fields_to_drop}
            m = m.select_entries(**new_entry_fields)

        return m

    @handle_py4j
    def filter_rows(self, expr):
        expr = to_expr(expr)
        base, cleanup = self._process_joins(expr)
        analyze(expr, self._row_indices, {self._col_axis}, set(self._fields.keys()))
        replace_aggregables(expr._ast, 'gs')
        m = Matrix(self._hc, base._jvds.filterVariantsExpr(expr._ast.to_hql(), True))
        return cleanup(m)

    @handle_py4j
    def filter_cols(self, expr):
        expr = to_expr(expr)
        base, cleanup = self._process_joins(expr)
        analyze(expr, self._col_indices, {self._row_axis}, set(self._fields.keys()))

        replace_aggregables(expr._ast, 'gs')
        m = Matrix(self._hc, base._jvds.filterSamplesExpr(expr._ast.to_hql(), True))
        return cleanup(m)

    @handle_py4j
    def filter_entries(self, expr):
        expr = to_expr(expr)
        base, cleanup = self._process_joins(expr)
        analyze(expr, self._entry_indices, set(), set(self._fields.keys()))

        m = Matrix(self._hc, base._jvds.filterGenotypes(expr._ast.to_hql(), True))
        return cleanup(m)

    def transmute_rows(self, **named_exprs):
        raise NotImplementedError()

    @handle_py4j
    def transmute_cols(self, **named_exprs):
        raise NotImplementedError()

    @handle_py4j
    def transmute_entries(self, **named_exprs):
        raise NotImplementedError()

    @handle_py4j
    def aggregate_rows(self, **named_exprs):
        str_exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, _ = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            allowed_fields = {'v', 'globals'}
            for f in self.row_schema.fields:
                allowed_fields.add(f.name)
            analyze(v, self._global_indices, {self._row_axis}, allowed_fields)
            replace_aggregables(v._ast, 'variants')
            str_exprs.append(v._ast.to_hql())

        result_list = self._jvds.queryVariants(jarray(Env.jvm().java.lang.String, str_exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    def aggregate_cols(self, **named_exprs):
        str_exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, _ = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            allowed_fields = {'s', 'globals'}
            for f in self.col_schema.fields:
                allowed_fields.add(f.name)
            analyze(v, self._global_indices, {self._col_axis}, allowed_fields)
            replace_aggregables(v._ast, 'samples')
            str_exprs.append(v._ast.to_hql())

        result_list = base._jvds.querySamples(jarray(Env.jvm().java.lang.String, str_exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    def aggregate_entries(self, **named_exprs):
        str_exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, _ = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze(v, self._global_indices, {self._row_axis, self._col_axis}, set(self._fields.keys()))
            replace_aggregables(v._ast, 'gs')
            str_exprs.append(v._ast.to_hql())

        result_list = base._jvds.queryGenotypes(jarray(Env.jvm().java.lang.String, str_exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    def explode_rows(self, expr):
        if isinstance(expr, str) or isinstance(expr, unicode):
            if not expr in self._fields:
                raise KeyError("Matrix has no field '{}'".format(expr))
            elif self._fields[expr].indices != self._row_indices:
                raise ExpressionException("Method 'explode_rows' expects a field indexed by row, found axes '{}'"
                                          .format(self._fields[expr].indices.axes))
            s = expr
        else:
            e = to_expr(expr)
            analyze(expr, self._row_indices, set(), set(self._fields.keys()))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_rows' requires a field or subfield, not a complex expression")
            s = e._ast.to_hql()

    @handle_py4j
    def explode_cols(self, expr):
        if isinstance(expr, str) or isinstance(expr, unicode):
            if not expr in self._fields:
                raise KeyError("Matrix has no field '{}'".format(expr))
            elif self._fields[expr].indices != self._col_indices:
                raise ExpressionException("Method 'explode_cols' expects a field indexed by col, found axes '{}'"
                                          .format(self._fields[expr].indices.axes))
            s = expr
        else:
            e = to_expr(expr)
            analyze(expr, self._col_indices, set(), set(self._fields.keys()))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_cols' requires a field or subfield, not a complex expression")
            s = e._ast.to_hql()

    @handle_py4j
    def group_rows_by(self, expr):
        expr = to_expr(expr)
        analyze(expr, self._row_indices, {self._col_axis}, set(self._fields.keys()))
        return GroupedMatrix(self, expr, self._row_indices)

    @handle_py4j
    def group_cols_by(self, expr):
        expr = to_expr(expr)
        analyze(expr, self._col_indices, {self._row_axis}, set(self._fields.keys()))
        return GroupedMatrix(self, expr, self._col_indices)

    @handle_py4j
    def count_rows(self):
        return self._jvds.countVariants()

    @handle_py4j
    @typecheck_method(output=strlike,
                      overwrite=bool)
    def write(self, output, overwrite=False):
        self._jvds.write(output, overwrite)

    @handle_py4j
    def rows_table(self):
        kt = hail2.Table(self._hc, self._jvds.variantsKT())

        # explode the 'va' struct to the top level
        return kt.select(kt.v, *kt.va)

    @handle_py4j
    def cols_table(self):
        kt = hail2.Table(self._hc, self._jvds.samplesKT())

        # explode the 'sa' struct to the top level
        return kt.select(kt.s, *kt.sa)

    @handle_py4j
    def entries_table(self):
        kt = hail2.Table(self._hc, self._jvds.genotypeKT())

        # explode the 'va', 'sa', 'g' structs to the top level
        # FIXME: this part should really be in Scala
        cols_to_select = tuple(x for x in kt.va) + tuple(x for x in kt.sa) + tuple(x for x in kt.g)
        return kt.select(kt.v, kt.s, *cols_to_select)

    @handle_py4j
    def index_globals(self):
        uid = Env._get_uid()

        def joiner(obj):
            if isinstance(obj, Matrix):
                return Matrix(obj._hc, Env.jutils().joinGlobals(obj._jvds, self._jvds, uid))
            else:
                from hail2.table import Table
                assert isinstance(obj, Table)
                return Table(obj._hc, Env.jutils().joinGlobals(obj._jkt, self._jvds, uid))

        return convert_expr(
            Expression(GlobalJoinReference, self.global_schema, joins=(Join(joiner, [uid]))))

    @handle_py4j
    def index_rows(self, expr):
        expr = to_expr(expr)
        indices, aggregations, joins = expr._indices, expr._aggregations, expr._joins
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        from hail2.table import Table
        if isinstance(src, Table):
            # join table with matrix.rows_table()
            right = self.rows_table()
            select_struct = Struct(**{k: right[k] for k in [f.name for f in self.row_schema.fields]})
            right = right.select(right.v, **{uid: select_struct})

            key_uid = Env._get_uid()
            uids_to_delete.append(key_uid)

            def joiner(left):
                pre_key = left.key
                left = Table(left._hc, left._jkt.annotate('{} = {}'.format(key_uid, expr._ast.to_hql())))
                left = left.key_by(key_uid)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                left = left.key_by(*pre_key)
                return left

            return convert_expr(
                Expression(Reference(uid), self.row_schema, indices, aggregations,
                           joins + (Join(joiner, uids_to_delete),)))
        else:
            assert isinstance(src, Matrix)

            # fast path
            if expr is src.v:
                prefix = 'va'
                joiner = lambda left: (
                    Matrix(left._hc,
                           left._jvds.annotateVariantsVDS(src._jvds, jsome('{}.{}'.format(prefix, uid)),
                                                          jnone())))
            elif indices == {'row'}:
                prefix = 'va'
                joiner = lambda left: (
                    Matrix(left._hc,
                           left._jvds.annotateVariantsTable(src._jvds.variantsKT(),
                                                            [expr._ast.to_hql()],
                                                            '{}.{}'.format(prefix, uid), None)))
            elif indices == {'column'}:
                prefix = 'sa'
                joiner = lambda left: (
                    Matrix(left._hc,
                           left._jvds.annotateSamplesTable(src._jvds.samplesKT(),
                                                           [expr._ast.to_hql()],
                                                           '{}.{}'.format(prefix, uid), None)))
            else:
                # FIXME: implement entry-based join in the expression language
                raise NotImplementedError('vds join with indices {}'.format(indices))

            return convert_expr(
                Expression(Select(Reference(prefix), uid),
                           self.row_schema, indices, aggregations, joins + (Join(joiner, uids_to_delete),)))

    @handle_py4j
    def index_cols(self, expr):
        expr = to_expr(expr)
        indices, aggregations, joins = expr._indices, expr._aggregations, expr._joins
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        from hail2.table import Table
        if isinstance(src, Table):
            # join table with matrix.cols_table()
            right = self.cols_table()
            select_struct = Struct(**{k: right[k] for k in [f.name for f in self.col_schema.fields]})
            right = right.select(right.s, **{uid: select_struct})

            key_uid = Env._get_uid()
            uids_to_delete.append(key_uid)

            def joiner(left):
                pre_key = left.key
                left = Table(left._hc, left._jkt.annotate('{} = {}'.format(key_uid, expr._ast.to_hql())))
                left = left.key_by(key_uid)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                left = left.key_by(*pre_key)
                return left

            return convert_expr(
                Expression(Reference(uid),
                           self.col_schema, indices, aggregations, joins + (Join(joiner, uids_to_delete),)))
        else:
            assert isinstance(src, Matrix)
            if indices == src._row_indices:
                prefix = 'sa'
                joiner = lambda left: (
                    Matrix(left._hc,
                           left._jvds.annotateSamplesTable(src._jvds.samplesKT(),
                                                           [expr._ast.to_hql()],
                                                           '{}.{}'.format(prefix, uid), None)))
            elif indices == src._col_indices:
                prefix = 'va'
                joiner = lambda left: (
                    Matrix(left._hc,
                           left._jvds.annotateVariantsTable(src._jvds.samplesKT(),
                                                            [expr._ast.to_hql()],
                                                            '{}.{}'.format(prefix, uid), None)))
            else:
                # FIXME: implement entry-based join in the expression language
                raise NotImplementedError('vds join with indices {}'.format(indices))
            return convert_expr(
                Expression(Select(Reference(prefix), uid),
                           self.col_schema, indices, aggregations, joins + (Join(joiner, uids_to_delete),)))

    @handle_py4j
    def index_entries(self, row_expr, col_expr):
        row_expr = to_expr(row_expr)
        col_expr = to_expr(col_expr)

        indices, aggregations, joins = unify_all(row_expr, col_expr)
        src = indices.source
        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        from hail2.table import Table
        if isinstance(src, Table):
            # join table with matrix.entries_table()
            right = self.entries_table()
            select_struct = Struct(**{k: right[k] for k in [f.name for f in self.entry_schema.fields]})
            right = right.select(right.v, right.s, **{uid: select_struct})

            row_key_uid = Env._get_uid()
            col_key_uid = Env._get_uid()
            uids_to_delete.append(row_key_uid)
            uids_to_delete.append(col_key_uid)

            def joiner(left):
                pre_key = left.key
                left = Table(left._hc, left._jkt.annotate('{} = {}, {} = {}'.format(
                    row_key_uid, row_expr._ast.to_hql(),
                    col_key_uid, col_expr._ast.to_hql())))
                left = left.key_by(row_key_uid, col_key_uid)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                left = left.key_by(*pre_key)
                return left

            return convert_expr(
                Expression(Reference(uid),
                           self.entry_schema, indices, aggregations, joins + (Join(joiner, uids_to_delete),)))
        else:
            raise NotImplementedError('matrix.index_entries with {}'.format(src.__class__))

    def to_hail1(self):
        import hail
        h1vds = hail.VariantDataset(self._hc, self._jvds)
        h1vds._set_history(History('is a mystery'))
        return h1vds

    @typecheck_method(name=strlike, indices=Indices)
    def _check_field_name(self, name, indices):
        if name in self._reserved:
            msg = 'name collision with reserved namespace: {}'.format(name)
            error('Analysis exception: {}'.format(msg))
            raise ExpressionException(msg)
        if name in set(self._fields.keys()) and not self._fields[name]._indices == indices:
            msg = 'name collision with field indexed by {}: {}'.format(indices, name)
            error('Analysis exception: {}'.format(msg))
            raise ExpressionException(msg)

    @typecheck_method(exprs=tupleof(Expression))
    def _process_joins(self, *exprs):

        all_uids = []
        left = self

        for e in exprs:
            rewrite_global_refs(e._ast, self)
            for j in e._joins:
                left = j.join_function(left)
                all_uids.extend(j.temp_vars)


        def cleanup(matrix):
            return matrix.drop(*all_uids)

        return left, cleanup
