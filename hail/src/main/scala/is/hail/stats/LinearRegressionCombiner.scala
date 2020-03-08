package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector, diag, inv}
import breeze.numerics.sqrt
import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types.physical.{PArray, PFloat64, PType}
import is.hail.expr.types.virtual._
import net.sourceforge.jdistlib.{F, T}

object LinearRegressionCombiner {
  val typ: Type = TStruct(
    "xty" -> TArray(TFloat64),
    "beta" -> TArray(TFloat64),
    "diag_inv" -> TArray(TFloat64),
    "beta0" -> TArray(TFloat64))
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
    val length = xType.loadLength(xOffset)
    assert(k == length)

    var i = 0
    while (i < length) {
      if (xType.isElementMissing(xOffset, i))
        return

      x(i) = Region.loadDouble(xType.loadElement(xOffset, i))
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

  def computeResult(): Option[(DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double])] = {
    if (n > k)      
      try {
        val b = xtx \ xty
        val diagInv = diag(inv(xtx))

        val xtx0 = xtx(0 until k0, 0 until k0)
        val xty0 = xty(0 until k0)
        val b0 = xtx0 \ xty0

        Some((xty, b, diagInv, b0))
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
      case Some((xty, b, diagInv, b0)) =>
        rvb.startArray(k) // xty
        var i = 0
        while (i < k) {
          rvb.addDouble(xty(i))
          i += 1
        }
        rvb.endArray()

        rvb.startArray(k) // beta
        i = 0
        while (i < k) {
          rvb.addDouble(b(i))
          i += 1
        }
        rvb.endArray()

        rvb.startArray(k) // diagInv
        i = 0
        while (i < k) {
          rvb.addDouble(diagInv(i))
          i += 1
        }
        rvb.endArray()

        rvb.startArray(k0) // b0
        i = 0
        while (i < k0) {
          rvb.addDouble(b0(i))
          i += 1
        }
        rvb.endArray()

      case None =>
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
        rvb.setMissing()
    }
    rvb.endStruct()
  }

  def result(): Annotation = {
    val result = computeResult()

    result match {
      case Some((xty, b, diagInv, b0)) =>
        Annotation(
          xty.toArray: IndexedSeq[Double],
          b.toArray: IndexedSeq[Double],
          diagInv.toArray: IndexedSeq[Double],
          b0.toArray: IndexedSeq[Double])
      case None =>
        Annotation(null, null, null, null)
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
