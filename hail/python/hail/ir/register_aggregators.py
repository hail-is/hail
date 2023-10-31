from .ir import register_aggregator


def register_aggregators():
    from hail.expr.types import dtype

    register_aggregator('ApproxCDF', (dtype('int32'),), (dtype('int32'),),
                        dtype('struct{levels:array<int32>,items:array<int32>,_compaction_counts:array<int32>}'))
    register_aggregator('ApproxCDF', (dtype('int32'),), (dtype('int64'),),
                        dtype('struct{levels:array<int32>,items:array<int64>,_compaction_counts:array<int32>}'))
    register_aggregator('ApproxCDF', (dtype('int32'),), (dtype('float32'),),
                        dtype('struct{levels:array<int32>,items:array<float32>,_compaction_counts:array<int32>}'))
    register_aggregator('ApproxCDF', (dtype('int32'),), (dtype('float64'),),
                        dtype('struct{levels:array<int32>,items:array<float64>,_compaction_counts:array<int32>}'))

    register_aggregator('Collect', (), (dtype("?in"),), dtype('array<?in>'))
    register_aggregator('Densify', (dtype('int32'),), (dtype("?in"),), dtype('?in'))

    info_score_aggregator_type = dtype('struct{score:float64,n_included:tint32}')
    register_aggregator('InfoScore', (), (dtype('array<float64>'),), info_score_aggregator_type)

    register_aggregator('Sum', (), (dtype('int64'),), dtype('int64'))
    register_aggregator('Sum', (), (dtype('float64'),), dtype('float64'))

    register_aggregator('Sum', (), (dtype('array<int64>'),), dtype('array<int64>'))
    register_aggregator('Sum', (), (dtype('array<float64>'),), dtype('array<float64>'))

    register_aggregator('CollectAsSet', (), (dtype("?in"),), dtype('set<?in>'))

    register_aggregator('Product', (), (dtype('int64'),), dtype('int64'))
    register_aggregator('Product', (), (dtype('float64'),), dtype('float64'))

    hwe_aggregator_type = dtype('struct { het_freq_hwe: float64, p_value: float64 }')
    register_aggregator('HardyWeinberg', (), (dtype('call'),), hwe_aggregator_type)

    register_aggregator('Max', (), (dtype('bool'),), dtype('bool'))
    register_aggregator('Max', (), (dtype('int32'),), dtype('int32'))
    register_aggregator('Max', (), (dtype('int64'),), dtype('int64'))
    register_aggregator('Max', (), (dtype('float32'),), dtype('float32'))
    register_aggregator('Max', (), (dtype('float64'),), dtype('float64'))

    register_aggregator('Min', (), (dtype('bool'),), dtype('bool'))
    register_aggregator('Min', (), (dtype('int32'),), dtype('int32'))
    register_aggregator('Min', (), (dtype('int64'),), dtype('int64'))
    register_aggregator('Min', (), (dtype('float32'),), dtype('float32'))
    register_aggregator('Min', (), (dtype('float64'),), dtype('float64'))

    register_aggregator('Count', (), (), dtype('int64'))

    register_aggregator('Counter', (), (dtype('?in'),), dtype('dict<?in, int64>'))

    register_aggregator('Take', (dtype('int32'),), (dtype('?in'),), dtype('array<?in>'))

    register_aggregator('ReservoirSample', (dtype('int32'),), (dtype('?in'),), dtype('array<?in>'))

    register_aggregator('TakeBy', (dtype('int32'),), (dtype('?in'), dtype('?key'),), dtype('array<?in>'))

    downsample_aggregator_type = dtype('array<tuple(float64, float64, array<str>)>')
    register_aggregator('Downsample', (dtype('int32'),), (dtype('float64'), dtype('float64'), dtype('array<?T>'),), downsample_aggregator_type)

    call_stats_aggregator_type = dtype('struct{AC: array<int32>,AF:array<float64>,AN:int32,homozygote_count:array<int32>}')
    register_aggregator('CallStats', (dtype('int32'),), (dtype('call'),), call_stats_aggregator_type)

    inbreeding_aggregator_type = dtype('struct{f_stat:float64,n_called:int64,expected_homs:float64,observed_homs:int64}')
    register_aggregator('Inbreeding', (), (dtype('call'), dtype('float64'),), inbreeding_aggregator_type)

    linreg_aggregator_type = dtype('struct{xty:array<float64>,beta:array<float64>,diag_inv:array<float64>,beta0:array<float64>}')
    register_aggregator('LinearRegression', (dtype('int32'), dtype('int32'),), (dtype('float64'), dtype('array<float64>'),), linreg_aggregator_type)

    register_aggregator('PrevNonnull', (), (dtype('?in'),), dtype('?in'))

    register_aggregator('ImputeType', (), (dtype('str'),),
                        dtype('struct{anyNonMissing: bool,'
                              'allDefined: bool,'
                              'supportsBool: bool,'
                              'supportsInt32: bool,'
                              'supportsInt64: bool,'
                              'supportsFloat64: bool}'))

    numeric_ndarray_type = dtype("ndarray<?T:numeric, ?nat>")
    register_aggregator('NDArraySum', (), (numeric_ndarray_type,), numeric_ndarray_type)
