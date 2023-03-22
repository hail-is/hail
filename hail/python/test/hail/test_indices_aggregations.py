import hail as hl


def test_array_slice_end():
    ht = hl.utils.range_matrix_table(1, 1)
    try:
        ht = ht.annotate_rows(c = hl.array([1,2,3])[:ht.col_idx])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_array_slice_start():
    ht = hl.utils.range_matrix_table(1, 1)
    try:
        ht = ht.annotate_rows(c = hl.array([1,2,3])[ht.col_idx:])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_array_slice_step():
    ht = hl.utils.range_matrix_table(1, 1)
    try:
        ht = ht.annotate_rows(c = hl.array([1,2,3])[::ht.col_idx])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_matmul():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_cols(a = hl.nd.array([0]))
    ht = ht.annotate_rows(b = hl.nd.array([0]))
    try:
        ht = ht.annotate_rows(c = ht.b @ ht.a)
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'a' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_index():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([0]))
    try:
        ht = ht.annotate_rows(c = ht.b[ht.col_idx])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_index_with_slice_1():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b[ht.col_idx, :])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_index_with_slice_2():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b[:, ht.col_idx])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_index_with_None_1():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b[ht.col_idx, None])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_index_with_None_2():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b[None, ht.col_idx])
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_reshape_1():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b.reshape((ht.col_idx, 1)))
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_reshape_2():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b.reshape((1, ht.col_idx)))
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'col_idx' (indices ['column'])" in exc.args[0]
    else:
        assert False


def test_ndarray_reshape_tuple():
    ht = hl.utils.range_matrix_table(1, 1)
    ht = ht.annotate_cols(a = hl.tuple((1, 1)))
    ht = ht.annotate_rows(b = hl.nd.array([[0]]))
    try:
        ht = ht.annotate_rows(c = ht.b.reshape(ht.a))
    except hl.ExpressionException as exc:
        assert 'scope violation' in exc.args[0]
        assert "'a' (indices ['column'])" in exc.args[0]
    else:
        assert False
