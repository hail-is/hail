import hail as hl

def gram(x):
    return x @ x.T

def pc_relate(mt, k):
    # scores is samples-by-k matrix
    _, scores, _ = hl.hwe_normalized_pca(mt.gt, k)

    # mean impute missing data
    mt = mt.select_entries(x = mt.gt.n_alt_alleles / 2)
    mt = mt.annotate_rows(mean = hl.agg.mean(mt.x))
    mt = mt.select_entries(x = hl.or_else(mt.x, mt.mean))
    g = mt.to_ndarray(mt.x)

    # X is k-by-variants matrix
    X = hl.lstsq(scores, g.T)

    mu = scores @ X

    centered = g - mu

    variance = mu * (1.0 - mu)
    stddev = hl.sqrt(variance)

    phi = gram(centered) / gram(stddev)

    # self_kin is a 1-tensor of length samples, it is a measure of how related
    # each sample is to itself under our model. The "inbreeding coefficient" is
    # given by (2 * self_kin - 1).
    self_kin = hl.diagonal(phi)

    dominance = (hl.case()
        .when(g == 0.0, mu)
        .when(g == 0.5, 0.0)
        .default(1.0 - mu))

    # I want to scale the variance (sample-by-variant) for each sample by the
    # sample's self-kinship. Numpy-style broadcasting requires that I use the
    # double transpose, thoughts?
    normalized_dominance = dominance - (variance.T * self_kin).T

    ibd2 = gram(normalized_dominance) / gram(variance)

    ibd0_for_close_kins = 1.0 - 4.0 * phi + k2

    # calculate identity-by-state-0
    ibs0 = hl.inner_product(
               g.T, g,
               lambda l, r: hl.agg.count(hl.abs(l - r) == 1.0))

    temp = (mu * mu) @ ((1.0 - mu) * (1.0 - mu))
    ibd0_for_distant_kins = ibs0 / (temp + temp.T)

    ibd0 = hl.cond(phi < pow(2, -5/2),
                   ibd0_for_close_kins,
                   ibd0_for_distant_kins)

    ibd1 = 1.0 - ibd2 - ibd0

    return (phi, ibd0, ibd1, ibd2)
