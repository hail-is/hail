import hail as hl


def pct(x):
    return f'{x*100:.2f}%'


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

    def append_agg(agg):
        return computations.append(agg)

    count = append_agg(hl.agg.count())
    to_print.append(('(Summary)', {'Number of records': lambda results: format(results[count])}))

    def recur_expr(expr, path):
        d = {}
        missingness = append_agg(hl.agg.count_where(hl.is_missing(expr)))
        d['type'] = lambda _: str(expr.dtype)
        d['missing'] = lambda \
                results: f'{results[missingness]} values ({pct(results[missingness] / results[count])})'

        t = expr.dtype

        if t in (hl.tint32, hl.tint64, hl.tfloat32, hl.tfloat64):
            stats = append_agg(hl.agg.stats(expr))
            if t in (hl.tint32, hl.tint64):
                d['minimum'] = lambda results: format(map_int(results[stats]['min']))
                d['maximum'] = lambda results: format(map_int(results[stats]['max']))
                d['sum'] = lambda results: format(map_int(results[stats]['sum']))
            else:
                d['minimum'] = lambda results: format(results[stats]['min'])
                d['maximum'] = lambda results: format(results[stats]['max'])
                d['sum'] = lambda results: format(results[stats]['sum'])
            d['mean'] = lambda results: format(results[stats]['mean'])
            d['stdev'] = lambda results: format(results[stats]['stdev'])
        elif t == hl.tbool:
            counter = append_agg(hl.agg.filter(hl.is_defined(expr), hl.agg.counter(expr)))
            d['counts'] = lambda results: format(results[counter])
        elif t == hl.tstr:
            size = append_agg(hl.agg.stats(hl.len(expr)))
            take = append_agg(hl.agg.filter(hl.is_defined(expr), hl.agg.take(expr, 5)))
            d['minimum size'] = lambda results: format(map_int(results[size]['min']))
            d['maximum size'] = lambda results: format(map_int(results[size]['max']))
            d['mean size'] = lambda results: format(results[size]['mean'])
            d['sample values'] = lambda results: format(results[take])
        elif t == hl.tcall:
            ploidy_counts = append_agg(hl.agg.filter(hl.is_defined(expr), hl.agg.counter(expr.ploidy)))
            phased_counts = append_agg(hl.agg.filter(hl.is_defined(expr), hl.agg.counter(expr.phased)))
            n_hom_ref = append_agg(hl.agg.count_where(expr.is_hom_ref()))
            n_hom_var = append_agg(hl.agg.count_where(expr.is_hom_var()))
            n_het = append_agg(hl.agg.count_where(expr.is_het()))
            d['homozygous reference'] = lambda results: format(results[n_hom_ref])
            d['heterozygous'] = lambda results: format(results[n_het])
            d['homozygous variant'] = lambda results: format(results[n_hom_var])
            d['ploidy'] = lambda results: format(results[ploidy_counts])
            d['phased'] = lambda results: format(results[phased_counts])
        elif isinstance(t, hl.tlocus):
            contig_counts = append_agg(hl.agg.filter(hl.is_defined(expr), hl.agg.counter(expr.contig)))
            d['contig counts'] = lambda results: format(results[contig_counts])
        elif isinstance(t, (hl.tset, hl.tdict, hl.tarray)):
            size = append_agg(hl.agg.stats(hl.len(expr)))
            d['minimum size'] = lambda results: format(map_int(results[size]['min']))
            d['maximum size'] = lambda results: format(map_int(results[size]['max']))
            d['mean size'] = lambda results: format(results[size]['mean'])
        to_print.append((path, d))
        if isinstance(t, hl.ttuple):
            for i in range(len(expr)):
                recur_expr(expr[i], f'{path} / {i}')
        if isinstance(t, hl.tstruct):
            for k, v in expr.items():
                recur_expr(v, f'{path} / {repr(k)[1:-1]}')

    if skip_top:
        for k, v in x.items():
            recur_expr(x[k], prefix + ' / ' + repr(k)[1:-1])
    else:
        recur_expr(x, prefix)

    return computations.result(), to_print
