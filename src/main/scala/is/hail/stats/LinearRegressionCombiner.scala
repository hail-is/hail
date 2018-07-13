package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, diag, dim, inv}
import breeze.numerics._
import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types._
import net.sourceforge.jdistlib.T

object LinearRegressionCombiner {
  val typ: Type = TStruct(
    "beta" -> TArray(TFloat64()),
    "standard_error" -> TArray(TFloat64()),
    "t_stat" -> TArray(TFloat64()),
    "p_value" -> TArray(TFloat64()),
    "n" -> TInt64())

  val xType = TArray(TFloat64())
}

class LinearRegressionCombiner(k: Int) extends Serializable {
  assert(k > 0)

  var n = 0L
  var x = new Array[Double](k)
  var xtx = DenseMatrix.zeros[Double](k, k)
  var xty = DenseVector.zeros[Double](k)
  var yty = 0.0

  val xType = LinearRegressionCombiner.xType

  def merge(y: Double, x: IndexedSeq[Double]) {
    assert(k == x.length)

    var j = 0
    while (j < k) {
      val xj = x(j)
      xty(j) += y * xj
      var i = 0
      while (i < k) {
        xtx(i, j) += x(i) * xj
        i += 1
      }
      j += 1
    }

    yty += y * y
    n += 1
  }

  def merge(region: Region, y: Double, xOffset: Long) {
    val length = xType.loadLength(region, xOffset)
    assert(k == length)

    var i = 0
    while (i < length) {
      if (xType.isElementMissing(region, xOffset, i))
        return

      x(i) = region.loadDouble(xType.loadElement(region, xOffset, i))
      i += 1
    }

    var j = 0
    while (j < k) {
      val xj = x(j)
      xty(j) += y * xj
      var i = 0
      while (i < k) {
        xtx(i, j) += x(i) * xj
        i += 1
      }
      j += 1
    }

    yty += y * y
    n += 1
  }

  def merge(other: LinearRegressionCombiner) {
    n += other.n
    xtx :+= other.xtx
    xty :+= other.xty
    yty += other.yty
  }

  def computeResult(): Option[(DenseVector[Double], DenseVector[Double], DenseVector[Double])] = {
    try {
      val b = xtx \ xty
      val rse2 = (yty - (xty dot b)) / (n - k) // residual standard error squared
      val se = sqrt(rse2 * diag(inv(xtx)))
      val t = b /:/ se
      Some((b, se, t))
    } catch {
      case e: breeze.linalg.MatrixSingularException => None
      case e: breeze.linalg.NotConvergedException => None
    }
  }

  def result(rvb: RegionValueBuilder) {
    val result = computeResult()

    rvb.startStruct()

    result match {
      case Some((b, se, t)) if n > k =>
        rvb.startArray(k) // beta
        var i = 0
          while (i < k) {
            rvb.addDouble(b(i))
            i += 1
          }
          rvb.endArray()

          rvb.startArray(k) // standard_error
          i = 0
          while (i < k) {
            rvb.addDouble(se(i))
            i += 1
          }
          rvb.endArray()

          rvb.startArray(k) // t_stat
          i = 0
          while (i < k) {
            rvb.addDouble(t(i))
            i += 1
          }
          rvb.endArray()

          rvb.startArray(k) // p_value
          i = 0
          while (i < k) {
            rvb.addDouble(2 * T.cumulative(-math.abs(t(i)), n - k, true, false))
            i += 1
          }
          rvb.endArray()
      case None =>
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
    }

    rvb.addLong(n) // n

    rvb.endStruct()

  }

  def result(): Annotation = {
    val result = computeResult()

    result match {
      case Some((b, se, t)) if n > k =>
        Annotation(b.toArray: IndexedSeq[Double],
          se.toArray: IndexedSeq[Double],
          t.toArray: IndexedSeq[Double],
          t.map(ti => 2 * T.cumulative(-math.abs(ti), n - k, true, false)).toArray: IndexedSeq[Double],
          n)
      case None =>
        Annotation(null, null, null, null, n)
    }
  }

  def clear() {
    n = 0
    x = new Array[Double](k)
    xtx = DenseMatrix.zeros[Double](k, k)
    xty = DenseVector.zeros[Double](k)
    yty = 0.0
  }

  def copy(): LinearRegressionCombiner = {
    val combiner = new LinearRegressionCombiner(k)
    combiner.n = n
    combiner.x = x.clone()
    combiner.xtx = xtx.copy
    combiner.xty = xty.copy
    combiner.yty = yty
    combiner
  }
}
