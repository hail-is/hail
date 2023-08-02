import hail as hl


def test_dsp_used_functions_have_not_moved():
    # DSP's variants team depends on these functions which are in experimental. Normally we do not
    # guarantee backwards compatibility but given the practical importance of these to production
    # pipelines at Broad, we ensure they continue to exist.

    assert hl.experimental.full_outer_join_mt is not None
