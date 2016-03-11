import numpy as np

def normalize(a):
    ms = np.mean(a, axis = 0, keepdims = True)
    return np.divide(np.subtract(a, ms), np.sqrt(2.0*np.multiply(ms/2.0, 1-ms/2.0)*a.shape[1]))

g = np.pad(np.diag([1.0,1,2]),((0,1),(0,0)),mode='constant')

print 'genotypes without a missing entry \n', g, '\n'

g[1,0] = 1.0/3

print 'genotypes with an mean-imputed missing entry \n', g, '\n'

n = normalize(g)

print 'normalized genotype matrix \n', n, '\n'

U, s, V = np.linalg.svd(n, full_matrices=0)

print 'scores \n', U.dot(np.diag(s)), '\n'
print 'loadings \n', V.transpose(), '\n'
print 'eigenvalues \n', np.multiply(s,s), '\n'
print 'roundoff error \n', U.dot(np.diag(s).dot(V)) - n, '\n'

print 'flattened arrays of scores, loadings and eigenvalues, respectively: \n'

print tuple(U.dot(np.diag(s)).transpose().flatten()), '\n'
print tuple(V.transpose().flatten()), '\n'
print tuple(np.multiply(s,s).flatten())