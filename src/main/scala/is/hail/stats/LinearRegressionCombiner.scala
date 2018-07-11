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
    "p_value" -> TArray(TFloat64()),
    "n" -> TInt64())

  val xsType = TArray(TFloat64())
}

class LinearRegressionCombiner(nxs: Int) extends Serializable {
  var n = 0L
  var xs = DenseVector.zeros[Double](nxs)
  var xtx = DenseMatrix.zeros[Double](nxs, nxs)
  var xty = DenseVector.zeros[Double](nxs)
  var yty = 0.0

  val xsType = LinearRegressionCombiner.xsType

  def merge(y: Double, xsArray: IndexedSeq[Double]) {
    assert(nxs == xsArray.length)
    xs = DenseVector(xsArray.toArray)

    n += 1
    xtx :+= xs * xs.t
    xty :+= xs * y
    yty += y * y
  }

  def merge(region: Region, y: Double, xsOffset: Long) {
    val length = xsType.loadLength(region, xsOffset)
    assert(nxs == length)

    var i = 0
    while (i < length) {
      if (xsType.isElementMissing(region, xsOffset, i))
        return

      xs(i) = region.loadDouble(xsType.loadElement(region, xsOffset, i))
      i += 1
    }

    n += 1
    xtx :+= xs * xs.t
    xty :+= xs * y
    yty += y * y
  }

  def merge(other: LinearRegressionCombiner) {
    n += other.n
    xtx :+= other.xtx
    xty :+= other.xty
    yty += other.yty
  }

  def computeResult(): (DenseVector[Double], DenseVector[Double], DenseVector[Double]) = {
    val b = xtx \ xty
    val rse2 = 1.0 / (n - nxs) * (yty - (xty dot b)) // residual standard error squared
    val se = sqrt(rse2 * diag(inv(xtx)))
    val t = b /:/ se
    (b, se, t)
  }

  def result(rvb: RegionValueBuilder) {
    val (b, se, t) = computeResult()

    rvb.startStruct()

    if (n != 0) {
      rvb.startArray(nxs) // beta
      var i = 0
      while (i < nxs) {
        rvb.addDouble(b(i))
        i += 1
      }
      rvb.endArray()
    } else
      rvb.setMissing()

    if (n != 0) {
      rvb.startArray(nxs) // standard_error
      var i = 0
      while (i < nxs) {
        rvb.addDouble(se(i))
        i += 1
      }
      rvb.endArray()
    } else
      rvb.setMissing()

    if (n != 0) {
      rvb.startArray(nxs) // p_value
      var i = 0
      while (i < nxs) {
        rvb.addDouble(2 * T.cumulative(-math.abs(t(i)), n - nxs, true, false))
        i += 1
      }
      rvb.endArray()
    } else
      rvb.setMissing()

    rvb.addLong(n) // n

    rvb.endStruct()

  }

  def result(): Annotation = {
    val (b, se, t) = computeResult()
    if (n != 0)
      Annotation(b.toArray: IndexedSeq[Double],
        se.toArray: IndexedSeq[Double],
        t.map(ti => 2 * T.cumulative(-math.abs(ti), n - nxs, true, false)).toArray: IndexedSeq[Double],
        n)
    else
      Annotation(null, null, null, n)
  }

  def clear() {
    n = 0
    xs = DenseVector.zeros[Double](nxs)
    xtx = DenseMatrix.zeros[Double](nxs, nxs)
    xty = DenseVector.zeros[Double](nxs)
    yty = 0.0
  }

  def copy(): LinearRegressionCombiner = {
    val combiner = new LinearRegressionCombiner(nxs)
    combiner.n = n
    combiner.xs = xs.copy
    combiner.xtx = xtx.copy
    combiner.xty = xty.copy
    combiner.yty = yty
    combiner
  }
}
