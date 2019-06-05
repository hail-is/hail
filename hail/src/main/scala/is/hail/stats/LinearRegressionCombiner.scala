package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, diag, inv}
import breeze.numerics._
import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types.physical.{PArray, PFloat64, PType}
import is.hail.expr.types.virtual._
import net.sourceforge.jdistlib.{F, T}

object LinearRegressionCombiner {
  val typ: Type = TStruct(
    "beta" -> TArray(TFloat64()),
    "standard_error" -> TArray(TFloat64()),
    "t_stat" -> TArray(TFloat64()),
    "p_value" -> TArray(TFloat64()),
    "multiple_standard_error" -> TFloat64(),
    "multiple_r_squared" -> TFloat64(),
    "adjusted_r_squared" -> TFloat64(),
    "f_stat" -> TFloat64(),
    "multiple_p_value" -> TFloat64(),
    "n" -> TInt64())
}

class LinearRegressionCombiner(k: Int, k0: Int, t: PType) extends Serializable {
  assert(k > 0)
  assert(k0 >= 0 && k0 <= k)
  assert(t.isOfType(PArray(PFloat64())))
  val xType = t.asInstanceOf[PArray]

  var n = 0L
  var x = new Array[Double](k)
  var xtx = DenseMatrix.zeros[Double](k, k)
  var xty = DenseVector.zeros[Double](k)
  var yty = 0.0

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

  def computeResult(): Option[(DenseVector[Double], DenseVector[Double], DenseVector[Double], Double, Double, Double, Double)] = {
    if (n > k)      
      try {
        val d = n - k
        val b = xtx \ xty
        val rss = yty - (xty dot b)
        val rse2 = rss / d // residual standard error squared
        val se = sqrt(rse2 * diag(inv(xtx)))
        val t = b /:/ se

        val xtx0 = xtx(0 until k0, 0 until k0)
        val xty0 = xty(0 until k0)
        val b0 = xtx0 \ xty0
        val rss0 = yty - (xty0 dot b0)
                
        val rse = math.sqrt(rse2)
        val r2 = 1 - rss / rss0
        val r2adj = 1 - (1 - r2) * (n - k0) / d
        val f = ((rss0 - rss) * d) / (rss * (k - k0))
        Some((b, se, t, rse, r2, r2adj, f))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    else
      None
  }

  def result(rvb: RegionValueBuilder) {
    val result = computeResult()

    rvb.startStruct()

    result match {
      case Some((b, se, t, rse, r2, r2adj, f)) =>
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
        
        rvb.addDouble(rse)
        rvb.addDouble(r2)
        rvb.addDouble(r2adj)
        rvb.addDouble(f)
        rvb.addDouble(F.cumulative(f, k - k0, n - k, false, false))
        
      case None =>
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
    }

    rvb.addLong(n)

    rvb.endStruct()
  }

  def result(): Annotation = {
    val result = computeResult()

    result match {
      case Some((b, se, t, rse, r2, r2adj, f)) =>
        Annotation(b.toArray: IndexedSeq[Double],
          se.toArray: IndexedSeq[Double],
          t.toArray: IndexedSeq[Double],
          t.map(ti => 2 * T.cumulative(-math.abs(ti), n - k, true, false)).toArray: IndexedSeq[Double],
          rse,
          r2,
          r2adj,
          f,
          F.cumulative(f, k - k0, n - k, false, false),
          n)
      case None =>
        Annotation(null, null, null, null, null, null, null, null, null, n)
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
    val combiner = new LinearRegressionCombiner(k, k0, xType)
    combiner.n = n
    combiner.x = x.clone()
    combiner.xtx = xtx.copy
    combiner.xty = xty.copy
    combiner.yty = yty
    combiner
  }
}
