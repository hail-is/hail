package is.hail.stats

import is.hail.utils._
import org.apache.spark.sql.Row

class DownsampleCombiner(nDivisions: Int) extends Serializable {
  type BinType = (Int, Int)
  type PointType = (Double, Double, Seq[String])

  var binToPoint: Map[BinType, PointType] = _
  var initPoint: PointType = _

  var left: Double = 0.0
  var right: Double = 0.0
  var bottom: Double = 0.0
  var top: Double = 0.0

  def toBinCoords(x: Double, y: Double): (Int, Int) = {
    val i = (((x - left) / (right - left)) * nDivisions).toInt
    val j = (((y - bottom) / (top - bottom)) * nDivisions).toInt
    (i, j)
  }

  def baseMerge(x: Double, y: Double, label: Seq[String]): DownsampleCombiner = {
    assert(binToPoint != null)
    binToPoint += (toBinCoords(x, y) -> ((x, y, label)))

    this
  }

  def merge(x: Double, y: Double, label: Seq[String]): DownsampleCombiner = {
    if (binToPoint != null) {
      val bins = toBinCoords(x, y)
      val xFactor = bins._1 / nDivisions
      val yFactor = bins._2 / nDivisions
      var collapse = false

      if (math.abs(xFactor) > 1) {
        collapse = true
        if (xFactor < 0)
          left = x
        else
          right = x
      }

      if (math.abs(yFactor) > 1) {
        collapse = true
        if (yFactor < 0)
          bottom = y
        else
          top = y
      }

      if (collapse) {
        val oldBinPoint = binToPoint
        binToPoint = Map.empty
        oldBinPoint.foreach { case (_, (xc, yc, l)) => baseMerge(xc, yc, l) }
      }

      baseMerge(x, y, label)
    } else if (initPoint != null) {
      left = math.min(initPoint._1, x)
      right = math.max(initPoint._1, x)
      bottom = math.min(initPoint._2, y)
      top = math.max(initPoint._2, y)

      binToPoint = Map.empty
      baseMerge(initPoint._1, initPoint._2, initPoint._3)
      baseMerge(x, y, label)
      initPoint = null
    } else
      initPoint = (x, y, label)

    this
  }

  def merge(that: DownsampleCombiner): DownsampleCombiner = {
    if (that.binToPoint != null)
      that.binToPoint.foreach { case (_, (x, y, l)) => merge(x, y, l) }
    else if (that.initPoint != null)
      merge(that.initPoint._1, that.initPoint._2, that.initPoint._3)

    this
  }

  def copy(): DownsampleCombiner = {
    val c = new DownsampleCombiner(nDivisions)
    if (binToPoint != null)
      c.binToPoint ++= binToPoint
    c.initPoint = initPoint

    c.left = left
    c.right = right
    c.bottom = bottom
    c.top = top

    c
  }

  def clear() = {
    binToPoint = null
    initPoint = null
    left = 0.0
    right = 0.0
    bottom = 0.0
    top = 0.0
  }

  def toRes: IndexedSeq[Row] = {
    if (binToPoint == null) {
      return IndexedSeq[Row]()
    }
    binToPoint.map { case (_, (x, y, l)) => Row(x, y, l) }.toFastIndexedSeq
  }
}
