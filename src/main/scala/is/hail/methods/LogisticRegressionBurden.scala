package is.hail.methods

import breeze.linalg._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

object LogisticRegressionBurden {
  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    aggExpr: String,
    test: String,
    yExpr: String,
    covExpr: Array[String]): (KeyTable, KeyTable) = {

    val logRegTest = LogisticRegressionTest.tests.getOrElse(test,
      fatal(s"Supported tests are ${ LogisticRegressionTest.tests.keys.mkString(", ") }, got: $test"))

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    if (completeSamplesSet(keyName))
      fatal(s"Key name '$keyName' clashes with a sample name")

    val logregFields = logRegTest.schema.asInstanceOf[TStruct].fields.map(_.name).toSet
    if (logregFields(keyName))
      fatal(s"Key name '$keyName' clashes with reserved $test logreg columns $logregFields")

    if (!y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all present values equal to 0 or 1")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    val nullModel = new LogisticRegressionModel(cov, y)
    val nullFit = nullModel.fit()

    if (!nullFit.converged)
      fatal("Failed to fit (unregulatized) logistic regression null model (covariates only): " + (
        if (nullFit.exploded)
          s"exploded at Newton iteration ${ nullFit.nIter }"
        else
          "Newton iteration failed to converge"))

    info(s"Aggregating variants by '$keyName' for $n samples...")

    def sampleKT = vds.filterSamples((s, sa) => completeSamplesSet(s))
      .aggregateBySamplePerVariantKey(keyName, variantKeys, aggExpr, singleKey)
      .cache()

    val keyType = sampleKT.fields(0).typ

    // d > 0 implies at least 1 sample
    val numericType = sampleKT.fields(1).typ

    if (!numericType.isInstanceOf[TNumeric])
      fatal(s"aggregate_expr type must be numeric, found $numericType")

    info(s"Running $test logistic regression burden test for ${sampleKT.count} keys on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val sc = sampleKT.hc.sc
    val yBc = sc.broadcast(y)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest)

    val (logregSignature, merger) = TStruct(keyName -> keyType).merge(logRegTest.schema.asInstanceOf[TStruct])

    val logregRDD = sampleKT.rdd.mapPartitions({ it =>
      val X = XBc.value.copy
      it.map { keyedRow =>
        X(::, -1) := RegressionUtils.keyedRowToVectorDouble(keyedRow)
        merger(Row(keyedRow.get(0)), logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation).asInstanceOf[Row]
      }
    })

    val logregKT = new KeyTable(sampleKT.hc, logregRDD, signature = logregSignature, key = Array(keyName))

    (logregKT, sampleKT)
  }
}