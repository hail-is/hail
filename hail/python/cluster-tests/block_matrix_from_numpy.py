import hail as hl
import numpy as np

x = np.random.randn(100, 100)
x_bm = hl.linalg.BlockMatrix.from_numpy(x, 8)
x_round_trip = x_bm.to_numpy()

assert np.allclose(x, x_round_trip)