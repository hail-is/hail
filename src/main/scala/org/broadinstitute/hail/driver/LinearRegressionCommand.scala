package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.{LinRegStats, LinearRegression}
import org.broadinstitute.hail.variant.Locus
import org.kohsuke.args4j.{Option => Args4jOption}

object LinearRegressionCommand extends Command {

  def name = "linreg"

  def description = "Compute beta, se, t, p-value with sample covariates"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-y", aliases = Array("--y"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"), usage = "Variant annotation root, a period-delimited path starting with `va'")
    var root: String = "va.linreg"
  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val pathVA = Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    def toDouble(t: BaseType, code: String): Any => Double = t match {
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

    val completeSampleSet = completeSamples.toSet
    val vdsForCompleteSamples = vds.filterSamples((s, sa) => completeSampleSet(s))

    val linreg = LinearRegression(vdsForCompleteSamples, y, cov)

    val (newVAS, inserter) = vdsForCompleteSamples.insertVA(LinRegStats.`type`, pathVA)

    state.copy(
      vds = vds.copy(
        rdd = vds.rdd.zipPartitions(linreg.rdd, preservesPartitioning = true) { case (it, jt) =>
          it.zip(jt).map { case ((v, (va, gs)), (v2, comb)) =>
            assert(v == v2)
            (v, (inserter(va, comb.map(_.toAnnotation)), gs))
          }
        }.toOrderedRDD(_.locus),
        vaSignature = newVAS
      )
    )
  }
}
