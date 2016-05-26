package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import scala.StringBuilder
import scala.collection.mutable
import scala.reflect.ClassTag

object LinRegStats {
  def `type`: Type = TStruct(
    ("nMissing", TInt),
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))
}

object LinRegUtils {
  def toDouble(t: BaseType, code: String): Any => Double = t match {
    case TInt => _.asInstanceOf[Int].toDouble
    case TLong => _.asInstanceOf[Long].toDouble
    case TFloat => _.asInstanceOf[Float].toDouble
    case TDouble => _.asInstanceOf[Double]
    case TBoolean => _.asInstanceOf[Boolean].toDouble
    case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
  }

  def prepareCovMatrixAndY(sampleIdsAndAnnotations: IndexedSeq[(String, Annotation)],
                           yName: String, covName: String, ec: EvalContext) = {
    val a = ec.a //FIXME: What is this used for?
    val sampleIds = sampleIdsAndAnnotations.map(_._1)

    val (yT, yQ) = Parser.parse(yName, ec)
    val yToDouble = toDouble(yT, yName)
    val ySA = sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      yQ().map(yToDouble)
    }

    val (covT, covQ) = Parser.parseExprs(covName, ec).unzip
    val covToDouble = (covT, covName.split(",").map(_.trim)).zipped.map(toDouble)
    val covSA = sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      (covQ.map(_()), covToDouble).zipped.map(_.map(_))
    }

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (ySA, covSA, sampleIds)
        .zipped
        .filter( (y, c, s) => y.isDefined && c.forall(_.isDefined))

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

    (completeSamples.toSet, y, cov)
  }
}

case class LinRegStats(nMissing: Int, beta: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(nMissing, beta, se, t, p)

  override def toString: String = {
    val sb = new StringBuilder()
    sb.append(nMissing)
    sb.append("\t")
    sb.append(beta)
    sb.append("\t")
    sb.append(se)
    sb.append("\t")
    sb.append(t)
    sb.append("\t")
    sb.append(p)
    sb.append("\n")
    sb.result()
  }
}

class LinRegBuilder extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var sparseLength = 0 // length of rowsX and valsX (ArrayBuilder has no length), used to track missingRowIndices
  private var sumX = 0d
  private var sumXX = 0d
  private var sumXY = 0d
  private var sumYMissing = 0d

  def merge(row: Int, d: Option[Double], y: DenseVector[Double]): LinRegBuilder = {
    d match {
      case Some(x) =>
        if (x != 0.0) {
          rowsX += row
          sparseLength += 1
          valsX += x
          sumX += x
          sumXX += x*x
          sumXY += x * y(row)
        }
      case None =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
        sumYMissing += y(row)
    }
    this
  }

  // variant is atomic => combOp merge not called
  def merge(that: LinRegBuilder): LinRegBuilder = {
    missingRowIndices ++= that.missingRowIndices.result().map(_ + sparseLength)
    rowsX ++= that.rowsX.result()
    valsX ++= that.valsX.result()
    sparseLength += that.sparseLength
    sumX += that.sumX
    sumXX += that.sumXX
    sumXY += that.sumXY
    sumYMissing += that.sumYMissing

    this
  }

  def stats(y: DenseVector[Double], n: Int): Option[(SparseVector[Double], Double, Double, Int)] = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = n - nMissing

    // all HomRef | all Het | all HomVar
    if (sumX == 0 || (sumX == nPresent && sumXX == nPresent) || sumX == 2 * nPresent)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      val x = new SparseVector[Double](rowsXArray, valsXArray, n)
      val xx = sumXX + meanX * meanX * nMissing
      val xy = sumXY + meanX * sumYMissing

      Some((x, xx, xy, nMissing))
    }
  }
}

object LinearRegression {
  def name = "LinearRegression"

  def apply[T](sc: SparkContext, y: DenseVector[Double], cov: Option[DenseMatrix[Double]],
            data: RDD[(T, Iterable[Option[Double]])])(implicit tct: ClassTag[T]): LinearRegression[T] = {

    require(cov.forall(_.rows == y.size))

    val n = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n - k - 2

    if (d < 1)
      fatal(s"$n samples and $k ${plural(k, "covariate")} with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n samples with $k sample ${plural(k, "covariate")}...")

    val covAndOnes: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n, 1))
      case None => DenseMatrix.ones[Double](n, 1)
    }

    val qt = qr.reduced.justQ(covAndOnes).t
    val qty = qt * y

    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    new LinearRegression(
      data.flatMapValues(_.zipWithIndex)
        .aggregateByKey[LinRegBuilder](new LinRegBuilder())(
          (lrb, v) => lrb.merge(v._2, v._1, yBc.value),
          (lrb1, lrb2) => lrb1.merge(lrb2))
          .mapValues { lrb: LinRegBuilder =>
            lrb.stats(yBc.value, n).map { stats => {
              val (x, xx, xy, nMissing) = stats

              val qtx = qtBc.value * x
              val qty = qtyBc.value
              val xxp: Double = xx - (qtx dot qtx)
              val xyp: Double = xy - (qtx dot qty)
              val yyp: Double = yypBc.value

              val b: Double = xyp / xxp
              val se = math.sqrt((yyp / xxp - b * b) / d)
              val t = b / se
              val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))

              LinRegStats(nMissing, b, se, t, p)}
          }
        }
      )
    }
}

case class LinearRegression[T](rdd: RDD[(T, Option[LinRegStats])])

