package is.hail.methods

import breeze.linalg._
import is.hail.annotations._
import is.hail.expr.types.TFloat64
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._

object LogisticRegression {

  def apply(vsm: MatrixTable,
    test: String,
    yField: String,
    xField: String,
    covFields: Array[String],
    root: String): MatrixTable = {
    val logRegTest = LogisticRegressionTest.tests(test)

    val (y, cov, completeColIdx) = RegressionUtils.getPhenoCovCompleteSamples(vsm, yField, covFields)

    if (!y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be bool or numeric with all present values equal to 0 or 1")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x and intercept) implies $d degrees of freedom.")

    info(s"logistic_regression: running $test on $n samples for response variable y,\n"
      + s"    with input variable x, intercept, and ${ k - 1 } additional ${ plural(k - 1, "covariate") }...")

    val nullModel = new LogisticRegressionModel(cov, y)
    var nullFit = nullModel.fit()

    if (!nullFit.converged)
      if (logRegTest == FirthTest)
        nullFit = LogisticRegressionFit(nullModel.bInterceptOnly(),
          None, None, 0, nullFit.nIter, exploded = nullFit.exploded, converged = false)
      else
        fatal("Failed to fit logistic regression null model (standard MLE with covariates only): " + (
          if (nullFit.exploded)
            s"exploded at Newton iteration ${ nullFit.nIter }"
          else
            "Newton iteration failed to converge"))

    val sc = vsm.sparkContext
    val completeColIdxBc = sc.broadcast(completeColIdx)

    val yBc = sc.broadcast(y)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest)

    val fullRowType = vsm.rvRowType
    val entryArrayType = vsm.matrixType.entryArrayType
    val entryType = vsm.entryType
    val fieldType = entryType.field(xField).typ

    assert(fieldType.isOfType(TFloat64()))

    val entryArrayIdx = vsm.entriesIndex
    val fieldIdx = entryType.fieldIdx(xField)

    val (newRVType, inserter) = vsm.rvRowType.unsafeStructInsert(logRegTest.schema, List(root))
    val newMatrixType = vsm.matrixType.copy(rvRowType = newRVType)

    val newRVD = vsm.rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType) { it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      val missingCompleteCols = new ArrayBuilder[Int]()

      val X = XBc.value.copy
      it.map { rv =>
        RegressionUtils.setMeanImputedDoubles(X.data, n * k, completeColIdxBc.value, missingCompleteCols, 
          rv, fullRowType, entryArrayType, entryType, entryArrayIdx, fieldIdx)

        val logregAnnot = logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation

        rvb.set(rv.region)
        rvb.start(newRVType)
        inserter(rv.region, rv.offset, rvb,
          () => rvb.addAnnotation(logRegTest.schema, logregAnnot))

        rv2.set(rv.region, rvb.end())
        rv2
      }
    }

    vsm.copyMT(matrixType = newMatrixType,
      rvd = newRVD)
  }
}
