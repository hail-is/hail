package is.hail.driver

import is.hail.utils._
import is.hail.stats._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.methods.LogisticRegression
import is.hail.stats.RegressionUtils.getPhenoCovCompleteSamples
import is.hail.variant.VariantDataset

object LogisticRegressionCommand {

  def tests = Map("wald" -> WaldTest, "lrt" -> LikelihoodRatioTest, "score" -> ScoreTest)

  def run(vds: VariantDataset, test: String, ySA: String, covSA: Array[String], root: String): VariantDataset = {

    val (y, cov, completeSamples) = getPhenoCovCompleteSamples(vds, ySA, covSA)

    if (! y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all values equal to 0 or 1")

    if (!tests.isDefinedAt(test))
      fatal(s"Supported tests are ${tests.keys.mkString(", ")}, got: $test")

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    LogisticRegression(vds, pathVA, completeSamples, y, cov, tests(test))
  }
}
