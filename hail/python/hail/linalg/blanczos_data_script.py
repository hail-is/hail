import numpy as np
import hail as hl
from hail import methods
import pandas as pd
from math import sqrt, pi
from random import randint, choice
import pandas as pd
import time

# Functions for operating with Tables of ndarrays in Hail (from Tim)

from hail.expr import Expression, ExpressionException, \
    expr_float64, expr_call, expr_any, expr_numeric, expr_array, \
    expr_locus, \
    analyze, check_entry_indexed, check_row_indexed, \
    matrix_table_source, table_source

# Only groups by rows, NOT COLUMNS
def matrix_table_to_table_of_ndarrays(field, group_size, tmp_path = '/tmp/nd_table.ht'):
    """

    The returned table has two fields: 'row_group_number' and 'ndarray'.

    Examples
    --------
    >>> ht = matrix_table_to_table_of_ndarrays(mt.GT.n_alt_alleles(), 100)

    Parameters
    ----------
    field
    group_size
    tmp_path

    Returns
    -------

    """
    mt = matrix_table_source('matrix_table_to_table_of_ndarrays/x', field)
    mt = mt.select_entries(x = field)
    ht = mt.localize_entries(entries_array_field_name='entries')
    # now ht.entries is an array of structs with one field, x

    # we'll also want to mean-impute/variance-normalize/etc here
    ht = ht.select(xs = ht.entries.map(lambda e: e['x']))
    # now ht.xs is an array of float64

    # now need to produce groups of G
    ht = ht.add_index()
    ht = ht.group_by(row_group_number= hl.int32(ht.idx // group_size)) \
        .aggregate(ndarray=hl.nd.array(hl.agg.collect(ht.xs)))
    # may require a .T on ndarray

    return ht.checkpoint(tmp_path, overwrite=True)

def chunk_ndarray(a, group_size):
    """Chunks a NDarray along the first axis in chunks of `group_size`.
    Parameters
    ----------
    a
    group_size
    -------

    """
    n_groups = a.shape[0] // group_size
    groups = []
    for i in range(a.shape[0] // group_size):
        start = i * group_size
        end = (i + 1) * group_size
        groups.append(a[start:end, :])
    return groups


# Concatenate the ndarrays with a blocked Table
def concatBlocked(A):
    blocks = A.ndarray.collect()
    big_mat = np.concatenate(blocks, axis=0)
    ht = ndarray_to_table([big_mat])
    
    block_shape = blocks[0].shape
    
    tup = ht.ndarray.collect()[0].shape
    assert tup == (len(blocks) * block_shape[0], block_shape[1])
    
    return ht

def concatToNumpy(A):
    blocks = A.ndarray.collect()
    big_mat = np.concatenate(blocks, axis=0)
    return big_mat

def ndarray_to_table(chunked_arr):
    structs = [hl.struct(row_group_number = idx, ndarray = block)
               for idx, block in enumerate(chunked_arr)]
    ht = hl.Table.parallelize(structs)
    ht = ht.key_by('row_group_number')
    return ht

# function to multiply two blocks, given the two blocks
# returns struct in form of array but not ndarray, includes the shape in the struct
# to change the result product directly back into a ndarray we need to use from_column_major
def block_product(left, right):
    product = left @ right
    n_rows, n_cols = product.shape
    return hl.struct(
        shape=product.shape,
        block=hl.range(hl.int(n_rows * n_cols)).map(
            lambda absolute: product[absolute % n_rows, absolute // n_rows]))

# takes in output of block_product
def block_aggregate(prod):
    shape = prod.shape
    block = prod.block
    return hl.nd.from_column_major(
        hl.agg.array_sum(block),
        hl.agg.take(shape, 1)[0])

# returns flat array
def to_column_major(ndarray):
    n_rows, n_cols = ndarray.shape
    return hl.range(hl.int(n_rows * n_cols)).map(
        lambda absolute: ndarray[absolute % n_rows, absolute // n_rows])



def matmul_rowblocked_nonblocked(A, B):
    temp = A.annotate_globals(mat = B)
    temp = temp.annotate(ndarray = temp.ndarray @ temp.mat)
    temp = temp.select(temp.ndarray)
    temp = temp.drop(temp.mat)
    return temp

def matmul_colblocked_rowblocked(A, B):
    temp = A.transmute(ndarray = block_product(A.ndarray.transpose(), B[A.row_group_number].ndarray))
    result_arr_sum = temp.aggregate(block_aggregate(temp.ndarray))
    return result_arr_sum

def computeNextH(A, H):
    nextG = matmul_colblocked_rowblocked(A, H)
    return matmul_rowblocked_nonblocked(A, nextG)

def hailBlanczos(A, G, m, n, k, l, q):
    
    # assert l > k
    # assert (q+1)*l <= (n - k)
    # assert n <= m
    
    start = time.time()
    
    Hi = matmul_rowblocked_nonblocked(A, G)
    npH = concatToNumpy(Hi)
    for j in range(0, q):
        Hj = computeNextH(A, Hi)
        npH = np.concatenate((npH, concatToNumpy(Hj)), axis=1)
        Hi = Hj
    
    # assert npH.shape == (m, (q+1)*l)
    # perform QR decomposition on unblocked version of H
    Q, R = np.linalg.qr(npH)
    # assert Q.shape == (m, (q+1)*l)
    
    # block Q's rows into the same number of blocks that A has
    num_blocks = A.count() # fix
    group_size_Q = Q.shape[0] // num_blocks
    #assert group_size_Q * num_blocks == m
    blocked_Q_table = ndarray_to_table(chunk_ndarray(Q, group_size_Q))
    
    T = matmul_colblocked_rowblocked(blocked_Q_table, A)
    # assert T.shape == ((q+1)*l, n)

    U, S, W = np.linalg.svd(T, full_matrices=False)
    # assert U.shape == ((q+1)*l, n)
    # assert S.shape == (n,)
    # assert W.shape == (n, n)
    
    sing_val = S[k]
    
    V = matmul_rowblocked_nonblocked(blocked_Q_table, U)
    arr_V = concatToNumpy(V)
    
    end = time.time()
    
    truncV = arr_V[:,:k]
    truncS = S[:k]
    truncW = W[:k,:]
    
    bound, satC = blanczosErrorB(truncV, np.diag(truncS), truncW.transpose(), m, n, k, q, concatToNumpy(A), sing_val)
    print("Satisfies Blanczos error bound equation 4.3 if C=1: ", bound)
    
    return truncV, truncS, truncW, sing_val, Q, bound, satC, end - start


def blanczosErrorA(U, S, V, m, n, k, q, A, k1th_sing_val):
    norm_diff = np.linalg.norm(A - U @ S @ V.transpose())
    bound = 100 * l * (((m-k)/l) ** (1/(4*q + 2))) * k1th_sing_val
    print('value:', norm_diff, 'bound/upper limit:', bound)
    return norm_diff <= bound

def blanczosErrorB(U, S, V, m, n, k, q, A, k1th_sing_val):
    C = 1
    norm_diff = np.linalg.norm(A - U @ S @ V.transpose())
    bound = C * (m ** (1/(4*q))) * k1th_sing_val
    satisfyingC = norm_diff / bound
    print('difference A - USV:', norm_diff, 'bound/upper limit:', bound)
    print('C constant needed to satisfy bound:', satisfyingC)
    return norm_diff <= bound, satisfyingC

def makeSharedData(model_input, block_size):
    
    # we should have m > n for hail implementation
    mt = hl.balding_nichols_model(*model_input)
    
    mt.write("balding_nichols_data.mt", overwrite=True)
    mt = hl.read_matrix_table("balding_nichols_data.mt")
    
    mt = mt.transmute_entries(n_alt = hl.float64(mt.GT.n_alt_alleles())) 
    table = matrix_table_to_table_of_ndarrays(mt.n_alt, block_size, tmp_path='/tmp/test_table.ht')
    
    # for numpy implementation we want transposed version so m < n
    np_matrix = np.asmatrix(concatToNumpy(table).transpose())

    return table, np_matrix



# SCRIPT:


df = pd.DataFrame(columns=['M', 'N', 'block size', 'K', 'L', 'Q', 'time', 'C'])

# references a dataframe df not passed into the function
def loop(i, m, n, block_size, k, l, q):

	#print('loop', i)

    try: 
        assert l > k
        assert n <= m
    except:
        return

    try:
        table, mat = makeSharedData((3, n, m), block_size)
    except:
        print('failed to make data with ', (m, n))
    
    try:
        G = hl.nd.array(np.random.normal(0, 1, (n,l)))
        _, S, _, _, _, _, C, time_passed = hailBlanczos(table, G, m, n, k, l, q)
        df.loc[i] = [m, n, block_size, k, l, q, time_passed, C]
        print(time_passed, 'seconds')
    except:
        print('failed during blanczos algorithm with ', (m, n))
        
    return


for i in range(1, 50):
    
    randN = 100 * randint(1, 100) #100
    randM = 100 * randint(1, 500) #1000
    randBlockSize = choice([2, 4, 10, 20, 25, 50])
    randK = randint(1, 100)
    randL = randint(1, 20) + randK
    randQ = randint(1, 5)
    
    loop(i, randM, randN, randBlockSize, randK, randL, randQ)
    
    df.to_csv('blanczos_data_BIG.csv')

