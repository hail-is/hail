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

    val completeSampleSet = completeSamples.toSet
    val vdsForCompleteSamples = vds.filterSamples((s, sa) => completeSampleSet(s))

    if (! y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all values equal to 0 or 1")

    if (!tests.isDefinedAt(test))
      fatal(s"Supported tests are ${tests.keys.mkString(", ")}, got: ${test}")

    val logreg = LogisticRegression(vdsForCompleteSamples, y, cov, tests(test))

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    val (newVAS, inserter) = vdsForCompleteSamples.insertVA(logreg.`type`, pathVA)

    vds.copy(
      rdd = vds.rdd.zipPartitions(logreg.rdd, preservesPartitioning = true) { case (it, jt) =>
        it.zip(jt).map { case ((v, (va, gs)), (v2, comb)) =>
          assert(v == v2)
          (v, (inserter(va, Some(comb)), gs))
        }
      }.asOrderedRDD,
      vaSignature = newVAS
    )
  }
}
