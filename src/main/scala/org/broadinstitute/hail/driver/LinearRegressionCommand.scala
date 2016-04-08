package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.methods.{CovariateData, LinRegStats, LinearRegression, Pedigree}
import org.kohsuke.args4j.{Option => Args4jOption}

object LinearRegressionCommand extends Command {

  def name = "linreg"

  def description = "Compute beta, std error, t-stat, and p-val for each SNP with additional sample covariates"

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-o", aliases = Array("--output"), usage = "Output root filename")
    var output: String = _

    @Args4jOption(required = true, name = "-y", aliases = Array("--y"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {

    // FIXME: catch input errors
    def parseSA(path: String): IndexedSeq[Option[Any]] = {
      state.vds.sampleAnnotations
        .map(
          state.vds.querySA(
            path.split(".").toList.tail
            ))
    }

    def makeDoubles(iseq: IndexedSeq[Option[Any]], mask: IndexedSeq[Boolean]): Array[Double] =
      iseq.zipWithIndex.filter(x => mask(x._2)).map(_._1.get.asInstanceOf[Double]).toArray

    val yA = parseSA(options.ySA)
    val covA = options.covSA.split(",").map(sa => parseSA(sa.trim()))

    // FIXME: could be function instead
    val sampleMask = Range(0, state.vds.nSamples).map(s => yA(s).isDefined && covA.forall(_(s).isDefined))

    val y = DenseVector(makeDoubles(yA, sampleMask))
    val cov = Some(new DenseMatrix(yA.size, covA.size, Array.concat(covA.map(makeDoubles(_, sampleMask)): _*)))

    val filtVds = state.vds.filterSamples((s, sa) => sampleMask(s))

    val linreg = LinearRegression(filtVds, y, cov)

    if (options.output != null)
      linreg.write(options.output)

    val (newVAS, inserter) = filtVds.insertVA(LinRegStats.`type`, "linreg")
    state.copy(
      vds = filtVds.copy(
        rdd = filtVds.rdd.zipPartitions(linreg.rdd) { case (it, jt) =>
          it.zip(jt).map { case ((v, va, gs), (v2, comb)) =>
            assert(v == v2)
            (v, inserter(va, comb.map(_.toAnnotation)), gs)
          }

        }, vaSignature = newVAS))
  }

}
