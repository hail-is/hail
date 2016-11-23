package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.LogisticRegression
import org.kohsuke.args4j.{Option => Args4jOption}

object LogisticRegressionCommand extends Command {

  def name = "logreg"

  def description = "Test each variant for association using the logistic regression model"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-t", aliases = Array("--test"), metaVar = "TEST",  usage = "Statistical test: `wald', `lrt', or `score'")
    var test: String = ""

    @Args4jOption(required = true, name = "-y", aliases = Array("--response"), metaVar = "Y", usage = "Response sample annotation, must be Boolean or numeric with all values 0 or 1")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), metaVar = "COV", usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"), usage = "Variant annotation root, a period-delimited path starting with `va'")
    var root: String = "va.logreg"
  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def tests = Map("wald" -> WaldTest, "lrt" -> LikelihoodRatioTest, "score" -> ScoreTest)

  def run(state: State, options: Options): State = {

    val vds = state.vds

    val pathVA = Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)
    val sb = new StringBuilder
    vds.saSignature.pretty(sb, 0, true)

    val a = ec.a

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
      a(0) = s
      a(1) = sa
      yQ().map(yToDouble)
    }

    val (covT, covQ) = Parser.parseExprs(options.covSA, ec).unzip
    val covToDouble = (covT, options.covSA.split(",").map(_.trim)).zipped.map(toDouble)
    val covSA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      (covQ.map(_()), covToDouble).zipped.map(_.map(_))
    }

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (ySA, covSA, vds.sampleIds)
        .zipped
        .filter( (y, c, s) => y.isDefined && c.forall(_.isDefined))

    val yArray = yForCompleteSamples.map(_.get).toArray
    val y = DenseVector(yArray)

    if (y.length == 0)
      fatal("Every sample in this logistic regression is missing at least one covariate or phenotype")

    if (! y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all values equal to 0 or 1")

    val sumY = sum(y)
    if (sumY == 0 || sumY == y.length)
      fatal(s"For logistic regression, phenotype cannot be constant, got all ${y(0)}")

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

    if (!tests.isDefinedAt(options.test))
      fatal(s"Supported tests are ${tests.keys.mkString(", ")}, got: ${options.test}")

    val logreg = LogisticRegression(vdsForCompleteSamples, y, cov, tests(options.test))

    val (newVAS, inserter) = vdsForCompleteSamples.insertVA(logreg.`type`, pathVA)

    state.copy(
      vds = vds.copy(
        rdd = vds.rdd.zipPartitions(logreg.rdd, preservesPartitioning = true) { case (it, jt) =>
          it.zip(jt).map { case ((v, (va, gs)), (v2, comb)) =>
            assert(v == v2)
            (v, (inserter(va, Some(comb)), gs))
          }
        }.asOrderedRDD,
        vaSignature = newVAS
      )
    )
  }
}
