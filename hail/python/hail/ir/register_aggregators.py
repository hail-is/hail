from .ir import register_aggregator

def register_aggregators():
    from hail.expr.types import dtype

    register_aggregator('ApproxCDF', (dtype('int32'),), None, (dtype('int32'),),
                        dtype('struct{values:array<int32>,ranks:array<int64>}'))
    register_aggregator('ApproxCDF', (dtype('int32'),), None, (dtype('int64'),),
                        dtype('struct{values:array<int64>,ranks:array<int64>}'))
    register_aggregator('ApproxCDF', (dtype('int32'),), None, (dtype('float32'),),
                        dtype('struct{values:array<float32>,ranks:array<int64>}'))
    register_aggregator('ApproxCDF', (dtype('int32'),), None, (dtype('float64'),),
                        dtype('struct{values:array<float64>,ranks:array<int64>}'))
    register_aggregator('Fraction', (), None, (dtype('bool'),), dtype('float64'))

    stats_aggregator_type = dtype('struct{mean:float64,stdev:float64,min:float64,max:float64,n:int64,sum:float64}')
    register_aggregator('Statistics', (), None, (dtype('float64'),), stats_aggregator_type)

    register_aggregator('Collect', (), None, (dtype("?in"),), dtype('array<?in>'))

    info_score_aggregator_type = dtype('struct{score:float64,n_included:tint32}')
    register_aggregator('InfoScore', (), None, (dtype('array<float64>'),), info_score_aggregator_type)

    register_aggregator('Sum', (), None, (dtype('int64'),), dtype('int64'))
    register_aggregator('Sum', (), None, (dtype('float64'),), dtype('float64'))

    register_aggregator('Sum', (), None, (dtype('array<int64>'),), dtype('array<int64>'))
    register_aggregator('Sum', (), None, (dtype('array<float64>'),), dtype('array<float64>'))

    register_aggregator('CollectAsSet', (), None, (dtype("?in"),), dtype('set<?in>'))

    register_aggregator('Product', (), None, (dtype('int64'),), dtype('int64'))
    register_aggregator('Product', (), None, (dtype('float64'),), dtype('float64'))

    hwe_aggregator_type = dtype('struct { het_freq_hwe: float64, p_value: float64 }')
    register_aggregator('HardyWeinberg', (), None, (dtype('call'),), hwe_aggregator_type)

    register_aggregator('Max', (), None, (dtype('bool'),), dtype('bool'))
    register_aggregator('Max', (), None, (dtype('int32'),), dtype('int32'))
    register_aggregator('Max', (), None, (dtype('int64'),), dtype('int64'))
    register_aggregator('Max', (), None, (dtype('float32'),), dtype('float32'))
    register_aggregator('Max', (), None, (dtype('float64'),), dtype('float64'))

    register_aggregator('Min', (), None, (dtype('bool'),), dtype('bool'))
    register_aggregator('Min', (), None, (dtype('int32'),), dtype('int32'))
    register_aggregator('Min', (), None, (dtype('int64'),), dtype('int64'))
    register_aggregator('Min', (), None, (dtype('float32'),), dtype('float32'))
    register_aggregator('Min', (), None, (dtype('float64'),), dtype('float64'))

    register_aggregator('Count', (), None, (), dtype('int64'))

    register_aggregator('Counter', (), None, (dtype('?in'),), dtype('dict<?in, int64>'))

    register_aggregator('Take', (dtype('int32'),), None, (dtype('?in'),), dtype('array<?in>'))

    register_aggregator('TakeBy', (dtype('int32'),), None, (dtype('?in'), dtype('?key'),), dtype('array<?in>'))

    histogram_aggregator_type = dtype('struct{bin_edges:array<float64>,bin_freq:array<int64>,n_smaller:int64,n_larger:int64}')
    register_aggregator('Histogram', (dtype('float64'), dtype('float64'), dtype('int32'),), None, (dtype('float64'),), histogram_aggregator_type)

    downsample_aggregator_type = dtype('array<tuple(float64, float64, array<str>)>')
    register_aggregator('Downsample', (dtype('int32'),), None, (dtype('float64'), dtype('float64'), dtype('array<?T>'),), downsample_aggregator_type)

    call_stats_aggregator_type = dtype('struct{AC: array<int32>,AF:array<float64>,AN:int32,homozygote_count:array<int32>}')
    register_aggregator('CallStats', (), (dtype('int32'),), (dtype('call'),), call_stats_aggregator_type)

    inbreeding_aggregator_type = dtype('struct{f_stat:float64,n_called:int64,expected_homs:float64,observed_homs:int64}')
    register_aggregator('Inbreeding', (), None, (dtype('call'), dtype('float64'),), inbreeding_aggregator_type)

    linreg_aggregator_type = dtype('struct{beta:array<float64>,standard_error:array<float64>,t_stat:array<float64>,p_value:array<float64>,multiple_standard_error:float64,multiple_r_squared:float64,adjusted_r_squared:float64,f_stat:float64,multiple_p_value:float64,n:int64}')
    register_aggregator('LinearRegression', (dtype('int32'), dtype('int32'),), None, (dtype('float64'), dtype('array<float64>'),), linreg_aggregator_type)

    register_aggregator('PearsonCorrelation', (), None, (dtype('tfloat64'), dtype('float64'),), dtype('float64'))

    register_aggregator('PrevNonnull', (), None, (dtype('?in'),), dtype('?in'))
