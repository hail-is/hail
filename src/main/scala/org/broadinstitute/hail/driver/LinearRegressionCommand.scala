package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.{LinRegStats, LinearRegression}
import org.kohsuke.args4j.{Option => Args4jOption}

object LinearRegressionCommand extends Command {

  def name = "linreg"

  def description = "Compute beta, se, t, p-value with sample covariates"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-y", aliases = Array("--y"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = false, name = "-o", aliases = Array("--output"), usage = "Path of output tsv")
    var output: String = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    def makeSampleMask(nSamples: Int, sa: IndexedSeq[Option[Any]]*): IndexedSeq[Boolean] = {
      require(sa.forall(_.size == nSamples))
      Range(0, nSamples).map(s => sa.forall(_(s).isDefined))
    }

    def makeArrayDouble(sa: IndexedSeq[Option[Any]], sampleMask: IndexedSeq[Boolean]): Array[Double] = {
      sa.iterator.zipWithIndex
        .filter(x => sampleMask(x._2))
        .map(_._1)
        .flatMap(_.map(_.asInstanceOf[Double]))
        .toArray
    }

    val vds = state.vds

    val qY = vds.querySA(Parser.parseAnnotationRoot(options.ySA, "sa"))
    val qCov = Parser.parseAnnotationRootList(options.covSA, "sa").map(vds.querySA)

    val ySA = vds.sampleAnnotations.map(qY)
    val covSA = qCov.map(q => vds.sampleAnnotations.map(q))

    val sampleMask = makeSampleMask(vds.nSamples, covSA :+ ySA: _*)

    val y = DenseVector(makeArrayDouble(ySA, sampleMask))
    val cov =
      if (covSA.isEmpty)
        None
      else
        Some(new DenseMatrix(
          y.size,
          covSA.size,
          Array.concat(covSA.map(makeArrayDouble(_, sampleMask)): _*)))

    val filtVds = vds.filterSamples((s, sa) => sampleMask(s))

    val linreg = LinearRegression(filtVds, y, cov)

    val (newVAS, inserter) = filtVds.insertVA(LinRegStats.`type`, "linreg")

    val newState = state.copy(
      vds = vds.copy(
        rdd = vds.rdd.zipPartitions(linreg.rdd) { case (it, jt) =>
          it.zip(jt).map { case ((v, va, gs), (v2, comb)) =>
            assert(v == v2)
            (v, inserter(va, comb.map(_.toAnnotation)), gs)
          }
        }, vaSignature = newVAS))

    if (options.output != null)
      ExportVariants.run(newState, Array("-o", options.output, "-c",
        """Chrom=v.contig,
          |Pos=v.start,
          |Ref=v.ref,
          |Alt=v.alt,
          |Missing=va.linreg.nMissing,
          |Beta=va.linreg.beta,
          |StdErr=va.linreg.se,
          |TStat=va.linreg.tstat,
          |PVal=va.linreg.pval""".stripMargin))

    newState
  }
}
