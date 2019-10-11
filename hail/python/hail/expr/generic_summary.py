import hail as hl
import functools


def pct(x):
    return f'{x * 100:.2f}%'


class Computations(object):
    def __init__(self):
        super(Computations, self).__init__()
        self.n = 0
        self.l = []

    def append(self, agg):
        self.l.append(agg)
        n_ret = self.n
        self.n += 1
        return n_ret

    def result(self):
        return hl.tuple(self.l)


def format(x):
    if isinstance(x, float):
        return f'{x:.2f}'
    else:
        return x


def map_int(x):
    if x is None:
        return x
    else:
        return int(x)


def generic_summary(x, prefix='', skip_top=False):
    computations = Computations()
    to_print = []

    def append_agg(c, agg):
        return c.append(agg)

    def index_into(r, p, i):
        x = r
        for idx in p:
            x = x[idx]
        return x[i]

    count = append_agg(computations, hl.agg.count())
    to_print.append(('(Summary)', {'Number of records': lambda results: format(index_into(results, (), count))}))

    def recur_expr(expr, context, path, c):
        d = {}
        missingness = append_agg(c, hl.agg.count_where(hl.is_missing(expr)))
        d['type'] = lambda _: str(expr.dtype)
        d['missing'] = lambda \
                results: f'{index_with_path(results, missingness)} values ({pct(index_with_path(results, missingness) / index_into(results, (), count))})'

        def index_with_path(r, i):
            return index_into(r, path, i)

        t = expr.dtype

        if t in (hl.tint32, hl.tint64, hl.tfloat32, hl.tfloat64):
            stats = append_agg(c, hl.agg.stats(expr))
            if t in (hl.tint32, hl.tint64):
                d['minimum'] = lambda results: format(map_int(index_with_path(results, stats)['min']))
                d['maximum'] = lambda results: format(map_int(index_with_path(results, stats)['max']))
                d['sum'] = lambda results: format(map_int(index_with_path(results, stats)['sum']))
            else:
                d['minimum'] = lambda results: format(index_with_path(results, stats)['min'])
                d['maximum'] = lambda results: format(index_with_path(results, stats)['max'])
                d['sum'] = lambda results: format(index_with_path(results, stats)['sum'])
            d['mean'] = lambda results: format(index_with_path(results, stats)['mean'])
            d['stdev'] = lambda results: format(index_with_path(results, stats)['stdev'])
        elif t == hl.tbool:
            counter = append_agg(c, hl.agg.filter(hl.is_defined(expr), hl.agg.counter(expr)))
            d['counts'] = lambda results: format(index_with_path(results, counter))
        elif t == hl.tstr:
            size = append_agg(c, hl.agg.stats(hl.len(expr)))
            take = append_agg(c, hl.agg.filter(hl.is_defined(expr), hl.agg.take(expr, 5)))
            d['minimum size'] = lambda results: format(map_int(index_with_path(results, size)['min']))
            d['maximum size'] = lambda results: format(map_int(index_with_path(results, size)['max']))
            d['mean size'] = lambda results: format(index_with_path(results, size)['mean'])
            d['sample values'] = lambda results: format(index_with_path(results, take))
        elif t == hl.tcall:
            n_hom_ref = append_agg(c, hl.agg.count_where(expr.is_hom_ref()))
            n_hom_var = append_agg(c, hl.agg.count_where(expr.is_hom_var()))
            n_het = append_agg(c, hl.agg.count_where(expr.is_het()))
            d['homozygous reference'] = lambda results: format(results[n_hom_ref])
            d['heterozygous'] = lambda results: format(results[n_het])
            d['homozygous variant'] = lambda results: format(results[n_hom_var])

            c2 = Computations()
            new_path = path + (c.n,)
            ploidy_counts = append_agg(c2, hl.agg.counter(expr.ploidy))
            phased_counts = append_agg(c2, hl.agg.counter(expr.ploidy))
            d['ploidy'] = lambda results: format(index_into(results, new_path, ploidy_counts))
            d['phased'] = lambda results: format(index_into(results, new_path, phased_counts))

            append_agg(c, hl.agg.filter(hl.is_defined(expr), c2.result()))
        elif isinstance(t, hl.tlocus):
            contig_counts = append_agg(c, hl.agg.filter(hl.is_defined(expr), hl.agg.counter(expr.contig)))
            d['contig counts'] = lambda results: format(index_with_path(results, contig_counts))
        elif isinstance(t, (hl.tset, hl.tdict, hl.tarray)):
            size = append_agg(c, hl.agg.stats(hl.len(expr)))
            d['minimum size'] = lambda results: format(map_int(results[size]['min']))
            d['maximum size'] = lambda results: format(map_int(results[size]['max']))
            d['mean size'] = lambda results: format(results[size]['mean'])
        to_print.append((context, d))
        if isinstance(t, hl.ttuple):
            for i in range(len(expr)):
                recur_expr(expr[i], f'{context}[{i}]', path, c)
        if isinstance(t, hl.tstruct):
            for k, v in expr.items():
                recur_expr(v, f'{context}[{repr(k)}]', path, c)
        if isinstance(t, (hl.tset, hl.tarray)):
            def explode_f(x):
                c2 = Computations()
                new_path = path + (c.n,)
                recur_expr(x, f'{context}[<elements>]', new_path, c2)
                return c2.result()

            append_agg(c, hl.agg.explode(explode_f, expr))
        if isinstance(t, hl.tdict):
            def explode_f(x):
                c2 = Computations()
                new_path = path + (c.n,)
                recur_expr(x[0], f'{context}[<keys>]', new_path, c2)
                recur_expr(x[1], f'{context}[<values>]', new_path, c2)
                return c2.result()

            append_agg(c, hl.agg.explode(explode_f, hl.array(expr)))

    if skip_top:
        for k, v in x.items():
            recur_expr(x[k], f'{prefix}[{repr(k)}]', (), computations)
    else:
        recur_expr(x, prefix, (), computations)

    return computations.result(), to_print
