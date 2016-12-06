package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.LinearRegression
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object LinearRegressionCommand extends Command {

  def name = "linreg"

  def description = "Test each variant for association using the linear regression model"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-y", aliases = Array("--response"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = false, name = "--mac", aliases = Array("--minimum-allele-count"), usage = "Minimum alternate allele count")
    var minAC: Int = 1

    @Args4jOption(required = false, name = "--maf", aliases = Array("--minimum-allele-freq"), usage = "Minimum alternate allele frequency")
    var minAF: java.lang.Double = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"), usage = "Variant annotation root, a period-delimited path starting with `va'")
    var root: String = "va.linreg"
  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (options.minAC < 1)
      fatal(s"Minumum alternate allele count must be a positive integer, got ${ options.minAC }")

    val minAF = Option(options.minAF).map(_.doubleValue())

    minAF.foreach { minAF => if (minAF <= 0d || minAF >= 1d)
      fatal(s"Minumum alternate allele frequency must lie in (0.0, 1.0), got ${ minAF }")
    }

    val pathVA = Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    def toDouble(t: Type, code: String): Any => Double = t match {
      case TInt => _.asInstanceOf[Int].toDouble
      case TLong => _.asInstanceOf[Long].toDouble
      case TFloat => _.asInstanceOf[Float].toDouble
      case TDouble => _.asInstanceOf[Double]
      case TBoolean => _.asInstanceOf[Boolean].toDouble
      case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
    }

    val (yT, yQ) = Parser.parse(options.ySA, ec)
    val yToDouble = toDouble(yT, options.ySA)
    val ySA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      yQ().map(yToDouble)
    }

    val (covT, covQ) = Parser.parseExprs(options.covSA, ec).unzip
    val covToDouble = (covT, options.covSA.split(",").map(_.trim)).zipped.map(toDouble)
    val covSA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      (covQ.map(_ ()), covToDouble).zipped.map(_.map(_))
    }

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (ySA, covSA, vds.sampleIds)
        .zipped
        .filter((y, c, s) => y.isDefined && c.forall(_.isDefined))

    val yArray = yForCompleteSamples.map(_.get).toArray
    val y = DenseVector(yArray)

    val covArray = covForCompleteSamples.flatMap(_.map(_.get)).toArray
    val k = covT.size
    val cov =
      if (k == 0)
        None
      else
        Some(new DenseMatrix(
          rows = completeSamples.size,
          cols = k,
          data = covArray,
          offset = 0,
          majorStride = k,
          isTranspose = true))

    val minAC = math.max(options.minAC, minAF.map(af => (math.ceil(2 * y.size * af) + 0.5).toInt).getOrElse(1))

    state.copy(vds = LinearRegression(vds, pathVA, completeSamples, y, cov, minAC))
  }
}
