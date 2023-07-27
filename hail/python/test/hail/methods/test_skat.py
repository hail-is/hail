import hail as hl
import pytest

from hail.utils import FatalError, HailUserError

from ..helpers import resource, test_timeout


@pytest.mark.parametrize("skat_model", [('hl._linear_skat', hl._linear_skat),
                                        ('hl._logistic_skat', hl._logistic_skat)])
def test_skat_negative_weights_errors(skat_model):
    skat_name, skat = skat_model
    genotypes = [
        [2, 1, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 1, 1, 0, 0],
        [0, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0]]
    covariates = [
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0]]
    phenotypes = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]
    weights = [-1, 0, 1, -1]

    mt = hl.utils.range_matrix_table(4, 15)
    mt = mt.annotate_entries(
        GT = hl.unphased_diploid_gt_index_call(
            hl.literal(genotypes)[mt.row_idx][mt.col_idx])
    )
    mt = mt.annotate_cols(
        phenotype = hl.literal(phenotypes)[mt.col_idx],
        cov1 = hl.literal(covariates)[mt.col_idx][0]
    )
    mt = mt.annotate_rows(
        weight = hl.literal(weights)[mt.row_idx]
    )
    mt = mt.annotate_globals(
        group = 0
    )
    ht = skat(mt.group, mt.weight, mt.phenotype, mt.GT.n_alt_alleles(), [1.0, mt.cov1])
    try:
        ht.collect()
    except Exception as exc:
        assert skat_name + ': every weight must be positive, in group 0, the weights were: [-1.0,0.0,1.0,-1.0]' in exc.args[0]
    else:
        assert False


def test_logistic_skat_phenotypes_are_binary():
    genotypes = [
        [2, 1, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 1, 1, 0, 0],
        [0, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0]]
    covariates = [
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0]]
    phenotypes = [0, 0, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]
    weights = [1, 1, 1, 1]

    mt = hl.utils.range_matrix_table(4, 15)
    mt = mt.annotate_entries(
        GT = hl.unphased_diploid_gt_index_call(
            hl.literal(genotypes)[mt.row_idx][mt.col_idx])
    )
    mt = mt.annotate_cols(
        phenotype = hl.literal(phenotypes)[mt.col_idx],
        cov1 = hl.literal(covariates)[mt.col_idx][0]
    )
    mt = mt.annotate_rows(
        weight = hl.literal(weights)[mt.row_idx]
    )
    mt = mt.annotate_globals(
        group = 0
    )
    try:
        ht = hl._logistic_skat(mt.group, mt.weight, mt.phenotype, mt.GT.n_alt_alleles(), [1.0, mt.cov1])
        ht.collect()
    except Exception as exc:
        assert 'hl._logistic_skat: phenotypes must either be True, False, 0, or 1, found: 3.0 of type float64' in exc.args[0]
    else:
        assert False


def test_logistic_skat_no_weights_R_truth():
    # library('SKAT')
    # dat <- matrix(c(2,1,0,1,
    #                 1,0,2,0,
    #                 1,1,0,0,
    #                 1,1,0,1,
    #                 0,1,2,1,
    #                 1,2,1,1,
    #                 1,0,1,1,
    #                 2,2nn,2,1,
    #                 1,1,2,1,
    #                 1,1,1,1,
    #                 2,0,1,1,
    #                 1,1,1,2,
    #                 0,1,0,2,
    #                 0,0,1,2,
    #                 1,0,1,0),
    #               15,
    #               4,
    #               byrow=TRUE)
    # cov <- data.frame(pheno=c(0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0),
    #                   cov1=c(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0))
    # weights <- c(1,1,1,1)
    # null_model <- SKAT_Null_Model(cov$pheno ~ cov$cov1, out_type="D", Adjustment=FALSE)
    # result <- SKAT(dat, null_model, method="davies", weights=weights)
    # cat(result$p.value, result$Q)
    expected_p_value = 0.5819739
    expected_Q_value = 1.869576

    genotypes = [
        [2, 1, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 1, 1, 0, 0],
        [0, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0]]
    covariates = [
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0]]
    phenotypes = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]
    weights = [1, 1, 1, 1]

    mt = hl.utils.range_matrix_table(4, 15)
    mt = mt.annotate_entries(
        GT = hl.unphased_diploid_gt_index_call(
            hl.literal(genotypes)[mt.row_idx][mt.col_idx])
    )
    mt = mt.annotate_cols(
        phenotype = hl.literal(phenotypes)[mt.col_idx],
        cov1 = hl.literal(covariates)[mt.col_idx][0]
    )
    mt = mt.annotate_rows(
        weight = hl.literal(weights)[mt.row_idx]
    )
    mt = mt.annotate_globals(
        group = 0
    )
    ht = hl._logistic_skat(mt.group, mt.weight, mt.phenotype, mt.GT.n_alt_alleles(), [1.0, mt.cov1])
    results = ht.collect()

    assert len(results) == 1
    result = results[0]
    assert result.size == 4
    assert result.q_stat == pytest.approx(expected_Q_value, abs=5e-7)
    assert result.p_value == pytest.approx(expected_p_value, abs=5e-8)
    assert result.fault == 0


def test_logistic_skat_R_truth():
    # library('SKAT')
    # dat <- matrix(c(2,1,0,1,
    #                 1,0,2,0,
    #                 1,1,0,0,
    #                 1,1,0,1,
    #                 0,1,2,1,
    #                 1,2,1,1,
    #                 1,0,1,1,
    #                 2,2,2,1,
    #                 1,1,2,1,
    #                 1,1,1,1,
    #                 2,0,1,1,
    #                 1,1,1,2,
    #                 0,1,0,2,
    #                 0,0,1,2,
    #                 1,0,1,0),
    #               15,
    #               4,
    #               byrow=TRUE)
    # cov <- data.frame(pheno=c(0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0),
    #                   cov1=c(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0))
    # # !!NOTA BENE!! In R, the "weights" parameter is actually the square root of the weights
    # weights <- c(1,sqrt(2),1,1)
    # null_model <- SKAT_Null_Model(cov$pheno ~ cov$cov1, out_type="D", Adjustment=FALSE)
    # result <- SKAT(dat, null_model, method="davies", weights=weights)
    # cat(result$p.value, result$Q)
    expected_p_value = 0.5335765
    expected_Q_value = 2.515238

    genotypes = [
        [2, 1, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 1, 1, 0, 0],
        [0, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0]]
    covariates = [
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0]]
    phenotypes = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]
    weights = [1, 2, 1, 1]

    mt = hl.utils.range_matrix_table(4, 15)
    mt = mt.annotate_entries(
        GT = hl.unphased_diploid_gt_index_call(
            hl.literal(genotypes)[mt.row_idx][mt.col_idx])
    )
    mt = mt.annotate_cols(
        phenotype = hl.literal(phenotypes)[mt.col_idx],
        cov1 = hl.literal(covariates)[mt.col_idx][0]
    )
    mt = mt.annotate_rows(
        weight = hl.literal(weights)[mt.row_idx]
    )
    mt = mt.annotate_globals(
        group = 0
    )
    ht = hl._logistic_skat(mt.group, mt.weight, mt.phenotype, mt.GT.n_alt_alleles(), [1.0, mt.cov1])
    results = ht.collect()

    assert len(results) == 1
    result = results[0]
    assert result.size == 4
    assert result.q_stat == pytest.approx(expected_Q_value, abs=5e-7)
    assert result.p_value == pytest.approx(expected_p_value, abs=5e-8)
    assert result.fault == 0


def test_logistic_skat_on_big_matrix():
    # dat <- as.matrix(read.csv('hail/src/test/resources/skat_genotype_matrix.csv', header=FALSE))
    # cov = read.csv('hail/src/test/resources/skat_phenotypes.csv', header=FALSE)
    # cov$V1 = cov$V1 > 2
    # weights <- rep(1, 100)
    # null_model <- SKAT_Null_Model(cov$V1 ~ 1, out_type="D")
    # result <- SKAT(dat, null_model, method="davies", weights=weights)
    #
    # cat(result$p.value, result$Q)
    #
    #
    # SKAT expects rows to be samples, so we transpose from the original input
    expected_p_value = 2.697155e-24
    expected_Q_value = 10046.37

    mt = hl.import_matrix_table(resource('skat_genotype_matrix_variants_are_rows.csv'),
                                delimiter=',',
                                row_fields={'row_idx': hl.tint64},
                                row_key=['row_idx'])
    mt = mt.key_cols_by(col_id=hl.int64(mt.col_id))

    ht = hl.import_table(resource('skat_phenotypes.csv'), no_header=True, types={'f0': hl.tfloat})
    ht = ht.add_index('idx')
    ht = ht.key_by('idx')
    mt = mt.annotate_cols(pheno=ht[mt.col_key].f0 > 2)
    mt = mt.annotate_globals(group=1)
    ht = hl._logistic_skat(mt.group, hl.literal(1), mt.pheno, mt.x, [1.0])
    results = ht.collect()

    assert len(results) == 1
    result = results[0]
    assert result.size == 100
    assert result.q_stat == pytest.approx(expected_Q_value, rel=5e-7)
    assert result.p_value == pytest.approx(expected_p_value, rel=5e-8)
    assert result.fault == 0


def test_linear_skat_no_weights_R_truth():
    # library('SKAT')
    # dat <- matrix(c(0,1,0,1,
    #                 1,0,1,0,
    #                 0,0,2,0,
    #                 0,0,0,2,
    #                 0,0,2,1),
    #               5,
    #               4,
    #               byrow=TRUE)
    # cov <- data.frame(pheno=c(3., 4., 6., 4., 1.), cov1=c(1.,3.,0.,6.,1.), cov2=c(2.,4.,9.,1.,1.))
    # weights <- c(1,1,1,1)
    # null_model <- SKAT_Null_Model(cov$pheno ~ cov$cov1+ cov$cov2, out_type="D", Adjustment=FALSE)
    # result <- SKAT(dat, null_model, method="davies", weights=weights)
    # cat(result$p.value, result$Q)
    expected_p_value = 0.2700286
    expected_Q_value = 2.854975

    genotypes = [
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 2, 0, 2],
        [1, 0, 0, 2, 1]]
    covariates = [
        [1, 2],
        [3, 4],
        [0, 9],
        [6, 1],
        [1, 1]]
    phenotypes = [3, 4, 6, 4, 1]
    weights = [1, 1, 1, 1]

    mt = hl.utils.range_matrix_table(4, 5)
    mt = mt.annotate_entries(
        GT = hl.unphased_diploid_gt_index_call(
            hl.literal(genotypes)[mt.row_idx][mt.col_idx])
    )
    mt = mt.annotate_cols(
        phenotype = hl.literal(phenotypes)[mt.col_idx],
        cov1 = hl.literal(covariates)[mt.col_idx][0],
        cov2 = hl.literal(covariates)[mt.col_idx][1]
    )
    mt = mt.annotate_rows(
        weight = hl.literal(weights)[mt.row_idx]
    )
    mt = mt.annotate_globals(
        group = 0
    )
    ht = hl._linear_skat(mt.group, mt.weight, mt.phenotype, mt.GT.n_alt_alleles(), [1.0, mt.cov1, mt.cov2])
    results = ht.collect()

    assert len(results) == 1
    result = results[0]
    assert result.size == 4
    assert result.q_stat == pytest.approx(expected_Q_value, abs=5e-7)
    assert result.p_value == pytest.approx(expected_p_value, abs=5e-8)
    assert result.fault == 0


def test_linear_skat_R_truth():
    # library('SKAT')
    # dat <- matrix(c(0,1,0,1,
    #                 1,0,1,0,
    #                 0,0,2,0,
    #                 0,0,0,2,
    #                 0,0,2,1),
    #               5,
    #               4,
    #               byrow=TRUE)
    # cov <- data.frame(pheno=c(3., 4., 6., 4., 1.), cov1=c(1.,3.,0.,6.,1.), cov2=c(2.,4.,9.,1.,1.))
    # # !!NOTA BENE!! In R, the "weights" parameter is actually the square root of the weights
    # weights <- c(1,sqrt(2),1,1)
    # null_model <- SKAT_Null_Model(cov$pheno ~ cov$cov1+ cov$cov2, out_type="C", Adjustment=FALSE)
    # result <- SKAT(dat, null_model, method="davies", weights=weights)
    # cat(result$p.value, result$Q)
    expected_p_value = 0.2497489
    expected_Q_value = 3.404505

    genotypes = [
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 2, 0, 2],
        [1, 0, 0, 2, 1]]
    covariates = [
        [1, 2],
        [3, 4],
        [0, 9],
        [6, 1],
        [1, 1]]
    phenotypes = [3, 4, 6, 4, 1]
    weights = [1, 2, 1, 1]

    mt = hl.utils.range_matrix_table(4, 5)
    mt = mt.annotate_entries(
        GT = hl.unphased_diploid_gt_index_call(
            hl.literal(genotypes)[mt.row_idx][mt.col_idx])
    )
    mt = mt.annotate_cols(
        phenotype = hl.literal(phenotypes)[mt.col_idx],
        cov1 = hl.literal(covariates)[mt.col_idx][0],
        cov2 = hl.literal(covariates)[mt.col_idx][1]
    )
    mt = mt.annotate_rows(
        weight = hl.literal(weights)[mt.row_idx]
    )
    mt = mt.annotate_globals(
        group = 0
    )
    ht = hl._linear_skat(mt.group, mt.weight, mt.phenotype, mt.GT.n_alt_alleles(), [1.0, mt.cov1, mt.cov2])
    results = ht.collect()

    assert len(results) == 1
    result = results[0]
    assert result.size == 4
    assert result.q_stat == pytest.approx(expected_Q_value, abs=5e-7)
    assert result.p_value == pytest.approx(expected_p_value, abs=5e-8)
    assert result.fault == 0


def _generate_skat_big_matrix():
    import numpy as np

    data = (np.random.random((2000, 100)) > 0.8) + 1.0 * (np.random.random((2000, 100)) > 0.8)
    with open(resource('skat_genotype_matrix.csv'), 'w') as f:
        for y in data:
            f.write(','.join([str(int(x)) for x in y]) + '\n')

    mt = hl.import_matrix_table(resource('skat_genotype_matrix.csv'), delimiter=',', no_header=True)
    bm = hl.linalg.BlockMatrix.from_entry_expr(mt.x)
    bm = bm.T
    mt = bm.to_matrix_table_row_major()
    mt = mt.annotate_entries(element=hl.int(mt.element))
    mt.element.export('skat_genotype_matrix_variants_are_rows.csv', delimiter=',')

    phenos = (data.sum(1) - 50) + np.random.normal(0, 10, 2000)
    with open(resource('skat_phenotypes.csv'), 'w') as f:
        for y in phenos:
            f.write(str(y) + '\n')


def test_linear_skat_on_big_matrix():
    # dat <- as.matrix(read.csv('hail/src/test/resources/skat_genotype_matrix.csv', header=FALSE))
    # cov = read.csv('hail/src/test/resources/skat_phenotypes.csv', header=FALSE)
    # weights <- rep(1, 100)
    # null_model <- SKAT_Null_Model(cov$V1 ~ 1, out_type="C")
    # result <- SKAT(dat, null_model, method="davies", weights=weights)
    #
    # cat(result$p.value, result$Q)
    #
    #
    # SKAT expects rows to be samples, so we transpose from the original input
    expected_p_value = 4.072862e-57
    expected_Q_value = 125247

    mt = hl.import_matrix_table(resource('skat_genotype_matrix_variants_are_rows.csv'),
                                delimiter=',',
                                row_fields={'row_idx': hl.tint64},
                                row_key=['row_idx'])
    mt = mt.key_cols_by(col_id=hl.int64(mt.col_id))

    ht = hl.import_table(resource('skat_phenotypes.csv'), no_header=True, types={'f0': hl.tfloat})
    ht = ht.add_index('idx')
    ht = ht.key_by('idx')
    mt = mt.annotate_cols(pheno=ht[mt.col_key].f0)
    mt = mt.annotate_globals(group=1)
    ht = hl._linear_skat(mt.group, hl.literal(1), mt.pheno, mt.x, [1.0])
    results = ht.collect()

    assert len(results) == 1
    result = results[0]
    assert result.size == 100
    assert result.q_stat == pytest.approx(expected_Q_value, rel=5e-7)
    assert result.p_value == pytest.approx(expected_p_value, rel=5e-8)
    assert result.fault == 0


def skat_dataset():
    ds2 = hl.import_vcf(resource('sample2.vcf'))

    covariates = (hl.import_table(resource("skat.cov"), impute=True)
                  .key_by("Sample"))

    phenotypes = (hl.import_table(resource("skat.pheno"),
                                  types={"Pheno": hl.tfloat64},
                                  missing="0")
                  .key_by("Sample"))

    intervals = (hl.import_locus_intervals(resource("skat.interval_list")))

    weights = (hl.import_table(resource("skat.weights"),
                               types={"locus": hl.tlocus(),
                                      "weight": hl.tfloat64})
               .key_by("locus"))

    ds = hl.split_multi_hts(ds2)
    ds = ds.annotate_rows(gene=intervals[ds.locus],
                          weight=weights[ds.locus].weight)
    ds = ds.annotate_cols(pheno=phenotypes[ds.s].Pheno,
                          cov=covariates[ds.s])
    ds = ds.annotate_cols(pheno=hl.if_else(ds.pheno == 1.0,
                                           False,
                                           hl.if_else(ds.pheno == 2.0,
                                                      True,
                                                      hl.missing(hl.tbool))))
    return ds


@test_timeout(3 * 60)
def test_skat_1():
    ds = skat_dataset()
    hl.skat(key_expr=ds.gene,
            weight_expr=ds.weight,
            y=ds.pheno,
            x=ds.GT.n_alt_alleles(),
            covariates=[1.0],
            logistic=False)._force_count()


@test_timeout(3 * 60)
def test_skat_2():
    ds = skat_dataset()
    hl.skat(key_expr=ds.gene,
            weight_expr=ds.weight,
            y=ds.pheno,
            x=ds.GT.n_alt_alleles(),
            covariates=[1.0],
            logistic=True)._force_count()

@test_timeout(3 * 60)
def test_skat_3():
    ds = skat_dataset()
    hl.skat(key_expr=ds.gene,
            weight_expr=ds.weight,
            y=ds.pheno,
            x=ds.GT.n_alt_alleles(),
            covariates=[1.0, ds.cov.Cov1, ds.cov.Cov2],
            logistic=False)._force_count()

@test_timeout(3 * 60)
def test_skat_4():
    ds = skat_dataset()
    hl.skat(key_expr=ds.gene,
            weight_expr=ds.weight,
            y=ds.pheno,
            x=hl.pl_dosage(ds.PL),
            covariates=[1.0, ds.cov.Cov1, ds.cov.Cov2],
            logistic=True)._force_count()

@test_timeout(3 * 60)
def test_skat_5():
    ds = skat_dataset()
    hl.skat(key_expr=ds.gene,
            weight_expr=ds.weight,
            y=ds.pheno,
            x=hl.pl_dosage(ds.PL),
            covariates=[1.0, ds.cov.Cov1, ds.cov.Cov2],
            logistic=(25, 1e-6))._force_count()


@test_timeout(local=4 * 60)
def test_linear_skat_produces_same_results_as_old_scala_method():
    mt = hl.import_vcf(resource('sample2.vcf'))
    covariates_ht = hl.import_table(
        resource("skat.cov"),
        key='Sample',
        types={'Cov1': hl.tint, 'Cov2': hl.tint}
    )
    phenotypes_ht = hl.import_table(
        resource("skat.pheno"),
        key='Sample',
        types={"Pheno": hl.tfloat64}, missing="0",
        impute=True
    )
    genes = hl.import_locus_intervals(
        resource("skat.interval_list")
    )
    weights = hl.import_table(
        resource("skat.weights"),
        key='locus',
        types={"locus": hl.tlocus(), "weight": hl.tfloat64}
    )
    mt = hl.split_multi_hts(mt)
    pheno = phenotypes_ht[mt.s].Pheno
    mt = mt.annotate_cols(
        cov = covariates_ht[mt.s],
        pheno = (hl.case()
                 .when(pheno == 1.0, False)
                 .when(pheno == 2.0, True)
                 .or_missing())
    )
    mt = mt.annotate_rows(
        gene = genes[mt.locus].target,
        weight = weights[mt.locus].weight
    )
    skat_results = hl._linear_skat(
        mt.gene,
        mt.weight,
        y=mt.pheno,
        x=mt.GT.n_alt_alleles(),
        covariates=[1, mt.cov.Cov1, mt.cov.Cov2]
    ).rename({'group': 'id'}).select_globals()
    old_scala_results = hl.import_table(
        resource('scala-skat-results.tsv'),
        types=dict(id=hl.tstr, size=hl.tint64, q_stat=hl.tfloat, p_value=hl.tfloat, fault=hl.tint),
        key='id'
    )
    assert skat_results._same(old_scala_results, tolerance=5e-5)  # TSV has 5 sigfigs, so we should match within 5e-5 relative



def test_skat_max_iteration_fails_explodes_in_37_steps():
    mt = hl.utils.range_matrix_table(3, 3)
    mt = mt.annotate_cols(y=hl.literal([1, 0, 1])[mt.col_idx])
    mt = mt.annotate_entries(
        x=hl.literal([
            [1, 0, 0],
            [10, 0, 0],
            [10, 5, 1]
        ])[mt.row_idx]
    )
    try:
        ht = hl.skat(
            hl.literal(0),
            mt.row_idx,
            y=mt.y,
            x=mt.x[mt.col_idx],
            logistic=(37, 1e-10),
            # The logistic settings are only used when fitting the null model, so we need to use a
            # covariate that triggers nonconvergence
            covariates=[mt.y]
        )
        ht.collect()[0]
    except FatalError as err:
        assert 'Failed to fit logistic regression null model (MLE with covariates only): exploded at Newton iteration 37' in err.args[0]
    except HailUserError as err:
        assert 'hl._logistic_skat: null model did not converge: {b: null, score: null, fisher: null, mu: null, n_iterations: 37, log_lkhd: -0.6931471805599453, converged: false, exploded: true}' in err.args[0]
    else:
        assert False


def test_skat_max_iterations_fails_to_converge_in_fewer_than_36_steps():
    mt = hl.utils.range_matrix_table(3, 3)
    mt = mt.annotate_cols(y=hl.literal([1, 0, 1])[mt.col_idx])
    mt = mt.annotate_entries(
        x=hl.literal([
            [1, 0, 0],
            [10, 0, 0],
            [10, 5, 1]
        ])[mt.row_idx]
    )
    try:
        ht = hl.skat(
            hl.literal(0),
            mt.row_idx,
            y=mt.y,
            x=mt.x[mt.col_idx],
            logistic=(36, 1e-10),
            # The logistic settings are only used when fitting the null model, so we need to use a
            # covariate that triggers nonconvergence
            covariates=[mt.y]
        )
        ht.collect()[0]
    except FatalError as err:
        assert 'Failed to fit logistic regression null model (MLE with covariates only): Newton iteration failed to converge' in err.args[0]
    except HailUserError as err:
        assert 'hl._logistic_skat: null model did not converge: {b: null, score: null, fisher: null, mu: null, n_iterations: 36, log_lkhd: -0.6931471805599457, converged: false, exploded: false}' in err.args[0]
    else:
        assert False
