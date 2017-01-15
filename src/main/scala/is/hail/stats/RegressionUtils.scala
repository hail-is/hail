package org.broadinstitute.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, rank}
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.VariantDataset
import org.broadinstitute.hail.utils._

object RegressionUtils {
  def toDouble(t: Type, code: String): Any => Double = t match {
    case TInt => _.asInstanceOf[Int].toDouble
    case TLong => _.asInstanceOf[Long].toDouble
    case TFloat => _.asInstanceOf[Float].toDouble
    case TDouble => _.asInstanceOf[Double]
    case TBoolean => _.asInstanceOf[Boolean].toDouble
    case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
  }

  def getPhenoCovCompleteSamples(
    vds: VariantDataset,
    ySA: String,
    covSA: Array[String]): (DenseVector[Double], DenseMatrix[Double], Array[Boolean]) = {

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    val (yT, yQ) = Parser.parseExpr(ySA, ec)
    val yToDouble = toDouble(yT, ySA)
    val yIS = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      yQ().map(yToDouble)
    }

    val (covT, covQ) = Parser.parseExprs(covSA.mkString(","), ec)
    val covToDouble = (covT, covSA).zipped.map(toDouble)
    val covIS = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      (covQ(), covToDouble).zipped.map(_.map(_))
    }

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (yIS, covIS, vds.sampleIds)
        .zipped
        .filter((y, c, s) => y.isDefined && c.forall(_.isDefined))

    val n = completeSamples.size
    if (n == 0)
      fatal("No complete samples: each sample is missing its phenotype or some covariate")

    val yArray = yForCompleteSamples.map(_.get).toArray
    if (yArray.toSet.size == 1)
      fatal(s"Constant phenotype: all complete samples have phenotype ${yArray(0)}")
    val y = DenseVector(yArray)

    val k = covT.size
    val covArray = covForCompleteSamples.flatMap(1.0 +: _.map(_.get)).toArray
    val cov = new DenseMatrix(
          rows = n,
          cols = 1 + k,
          data = covArray,
          offset = 0,
          majorStride = 1 + k,
          isTranspose = true)
    val r = rank(cov)
    if (r < k)
      fatal(s"Covariates and intercept are not linearly independent: total rank is $r (less than 1 + $k)")

    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray

    assert(completeSamplesSet.size == completeSamples.size)
    assert(y.size == completeSamples.size)
    assert(cov.rows == completeSamples.size)

    (y, cov, sampleMask)
  }

  def buildGtColumn(gts: Iterable[Option[Int]]): Option[DenseMatrix[Double]] = {
    val (nCalled, gtSum, allHet) = gts.flatten.foldLeft((0, 0, true))((acc, gt) => (acc._1 + 1, acc._2 + gt, acc._3 && (gt == 1) ))

    // allHomRef || allHet || allHomVar || allNoCall
    if (gtSum == 0 || allHet || gtSum == 2 * nCalled || nCalled == 0 )
      None
    else {
      val gtMean = gtSum.toDouble / nCalled
      val gtArray = gts.map(_.map(_.toDouble).getOrElse(gtMean)).toArray
      Some(new DenseMatrix(gtArray.length, 1, gtArray))
    }
  }
}