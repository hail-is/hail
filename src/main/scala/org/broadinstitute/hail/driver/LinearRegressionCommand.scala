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

    def maskSeq[T](xs: IndexedSeq[T], sampleMask: IndexedSeq[Boolean]): IndexedSeq[T] =
      xs.zip(sampleMask)
        .filter(_._2)
        .map(_._1)

    def toDouble(t: BaseType): Any => Double = (t: @unchecked) match {
      case TInt => _.asInstanceOf[Int].toDouble
      case TLong => _.asInstanceOf[Long].toDouble
      case TFloat => _.asInstanceOf[Float].toDouble
      case TDouble => _.asInstanceOf[Double]
      case TBoolean => _.asInstanceOf[Boolean].toDouble
      case _ => fatal(s"??? expected Numeric or Boolean, got `$t'")
    }

    val vds = state.vds

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val sb = new StringBuilder
    vds.saSignature.pretty(sb, 0, true)

    val a = new ArrayBuffer[Any]()
    for (_ <- symTab)
      a += null

    val yE = Parser.parse(options.ySA, symTab, a)
    val yToDouble = toDouble(yE._1)
    val ySA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      yE._2().map(yToDouble)
    }
    val yMask = ySA.map(_.isDefined)

    val covE = Parser.parseExprs(options.covSA, symTab, a)
    val covToDouble = covE.map(x => toDouble(x._1))
    val covSA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      (covE.map(_._2()), covToDouble).zipped.map( _.map(_))
    }
    val covMask = covSA.map(_.forall(_.isDefined))

    val sampleMask = (yMask, covMask).zipped.map(_ && _)

    val y = DenseVector(ySA
        .zip(sampleMask)
        .filter(_._2)
        .flatMap(_._1)
        .toArray)

    println(y)

    val covArrays = covSA
      .zip(sampleMask)
      .filter(_._2)
      .map(_._1.map(_.get))

    covArrays.foreach(a => println(a.mkString(",")))

    val cov =
      if (covArrays(0).isEmpty)
        None
      else
        Some(new DenseMatrix(
          rows = covArrays.size,
          cols = covArrays(0).size,
          data = Array.concat(covArrays: _*),
          offset = 0,
          majorStride = covArrays(0).size,
          isTranspose = true))

    val sampleIndex = vds.sampleIds.zipWithIndex.toMap

    val filtVds = vds.filterSamples((s, sa) => sampleMask(sampleIndex(s)))

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
