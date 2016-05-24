package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.{LinRegStats, LinearRegression, LinRegUtils}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.variant.Variant

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
    val sb = new StringBuilder
    vds.saSignature.pretty(sb, 0, true)

    val (completeSampleSet, y, cov) = LinRegUtils.prepareCovMatrixAndY(vds.sampleIdsAndAnnotations, options.ySA, options.covSA, ec)
    val vdsForCompleteSamples = vds.filterSamples((s, sa) => completeSampleSet(s))
    val genotypeData = vdsForCompleteSamples.rdd.map{case (v, va, gs) => (v, gs.map(_.nNonRefAlleles.map(_.toDouble)))}

    val linreg = LinearRegression[Variant](state.sc, y, cov, genotypeData)

    val (newVAS, inserter) = vdsForCompleteSamples.insertVA(LinRegStats.`type`, pathVA)

    state.copy(
      vds = vds.copy(
        rdd = vds.rdd.map{case (v, va, gs) => (v, (va, gs))}.fullOuterJoin(linreg.rdd).map {case (v, (x, y)) =>
          val (va, gs) = x.getOrElse(throw new IllegalArgumentException("join of rdd and linreg results didn't work"))
          val results = y.getOrElse(throw new IllegalArgumentException("join of rdd and linreg results didn't work"))
          (v, inserter(va, results.map(_.toAnnotation)), gs)
        },
        vaSignature = newVAS
      )
    )
  }
}
