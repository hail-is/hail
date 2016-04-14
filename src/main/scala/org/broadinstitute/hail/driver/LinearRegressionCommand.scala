package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.{LinRegStats, LinearRegression}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable.ArrayBuffer

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

    val vds = state.vds

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val sb = new StringBuilder
    vds.saSignature.pretty(sb, 0, true)

    val a = new ArrayBuffer[Any]()
    for (_ <- symTab)
      a += null

    def toDouble(t: BaseType, code: String): Any => Double = (t: @unchecked) match {
      case TInt => _.asInstanceOf[Int].toDouble
      case TLong => _.asInstanceOf[Long].toDouble
      case TFloat => _.asInstanceOf[Float].toDouble
      case TDouble => _.asInstanceOf[Double]
      case TBoolean => _.asInstanceOf[Boolean].toDouble
      case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
    }

    val yQ = Parser.parse(options.ySA, symTab, a)
    val yToDouble = toDouble(yQ._1, options.ySA)
    val ySA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      yQ._2().map(yToDouble)
    }

    val covQ = Parser.parseExprs(options.covSA, symTab, a)
    val covToDouble = (covQ.map(_._1), Parser.parseCodes(options.covSA)).zipped.map(toDouble)
    val covSA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      (covQ.map(_._2()), covToDouble).zipped.map(_.map(_))
    }

    val (yFilt, covFilt, sampleIdsFilt) =
      (ySA, covSA, vds.sampleIds)
        .zipped
        .filter( (y, c, s) => y.isDefined && c.forall(_.isDefined))

    val yArray = yFilt.map(_.get).toArray
    val y = DenseVector(yArray)

    val covArrays = covFilt.map(_.map(_.get))
    val n = covArrays.size
    val k = covArrays(0).size
    val cov =
      if (k == 0)
        None
      else
        Some(new DenseMatrix(
          rows = n,
          cols = k,
          data = Array.concat(covArrays: _*),
          offset = 0,
          majorStride = k,
          isTranspose = true))

    val sampleIdsFiltSet = sampleIdsFilt.toSet
    val vdsFilt = vds.filterSamples((s, sa) => sampleIdsFiltSet(s))

    val linreg = LinearRegression(vdsFilt, y, cov)

    val (newVAS, inserter) = vdsFilt.insertVA(LinRegStats.`type`, "linreg")

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
