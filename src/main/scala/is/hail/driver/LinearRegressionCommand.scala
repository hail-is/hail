package is.hail.driver

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.methods.LinearRegression
import is.hail.utils._
import is.hail.variant.VariantDataset
import is.hail.stats.RegressionUtils._

object LinearRegressionCommand {

  def run(vds: VariantDataset, ySA: String, covSA: Array[String], root: String, minAC: Int, minAF: Double): VariantDataset = {

    val (y, cov, completeSamples) = getPhenoCovCompleteSamples(vds, ySA, covSA)

    if (minAC < 1)
      fatal(s"Minumum alternate allele count must be a positive integer, got $minAC")
    if (minAF < 0d || minAF > 1d)
      fatal(s"Minumum alternate allele frequency must lie in [0.0, 1.0], got $minAF")
    val combinedMinAC = math.max(minAC, (math.ceil(2 * y.size * minAF) + 0.5).toInt)

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    LinearRegression(vds, pathVA, completeSamples, y, cov, combinedMinAC)
  }
}
