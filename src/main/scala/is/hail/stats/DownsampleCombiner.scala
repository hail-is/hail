package is.hail.stats

import org.apache.spark.sql.Row
import is.hail.utils._

class DownsampleCombiner(nDivisions: Int) extends Serializable {
  var binsToCoords: Map[(Int, Int), (Double, Double)] = Map.empty[(Int, Int), (Double, Double)]
  var initFrame: IndexedSeq[(Double, Double)] = FastIndexedSeq.empty[(Double, Double)]
  var hasPoints: Boolean = false

  var left: Double = 0.0
  var right: Double = 0.0
  var bottom: Double = 0.0
  var top: Double = 0.0

  def collapse(binsToCoords: Map[(Int, Int), (Double, Double)], nDivisions: Int,
               factor: Int, collapseX: Boolean, collapseY: Boolean): Map[(Int, Int), (Double, Double)] = {
    val mappings: Map[Int, Int] =
      Array.range(-nDivisions * math.pow(factor, 2).toInt, nDivisions * (-math.pow(factor, 2).toInt + math.pow(3, factor).toInt))
        .zip(0.until(nDivisions).flatMap(e => Array.fill(math.pow(3, factor).toInt)(e)))
        .toMap

    binsToCoords.map { case ((i, j), (x, y)) => (if (collapseX) mappings(i) else i, if (collapseY) mappings(j) else j) -> (x, y) }
  }

  def toBinCoords(x: Double, y: Double, left: Double, right: Double, bottom: Double, top: Double, nDivisions: Int): (Int, Int) = {
    val i = (((x - left) / (right - left)) * nDivisions).toInt
    val j = (((y - bottom) / (top - bottom)) * nDivisions).toInt
    (i, j)
  }

  def buildFactor(coord: Double, lower: Double, upper: Double, factor: Int = 0): Int = {
    if (coord < lower || coord > upper)
      buildFactor(coord, lower - (upper - lower), upper + (upper - lower), factor + 1)
    else
      factor
  }

  def merge(x: Double, y: Double): DownsampleCombiner = {
    if (!hasPoints) {
      initFrame :+= (x, y)

      if (initFrame.length == 2) {
        hasPoints = true

        left = initFrame.minBy(_._1)._1
        right = initFrame.maxBy(_._1)._1
        bottom = initFrame.minBy(_._2)._2
        top = initFrame.maxBy(_._2)._2

        initFrame.foreach { case (xc, yc) => binsToCoords += (toBinCoords(xc, yc, left, right, bottom, top, nDivisions) -> (xc, yc)) }
        initFrame = FastIndexedSeq.empty
      }
    } else {
      val xFactor = buildFactor(x, left, right)
      val yFactor = buildFactor(y, bottom, top)

      if (xFactor > 1 && yFactor > 1) {
        0.until(math.max(xFactor, yFactor)).foreach(_ => {
          val xDist = right - left
          val yDist = top - bottom
          left -= xDist
          right += xDist
          bottom -= yDist
          top += yDist
        })

        binsToCoords = collapse(binsToCoords, nDivisions, math.max(xFactor, yFactor), collapseX = true, collapseY = true)
      } else if (xFactor > 1) {
        0.until(xFactor).foreach(_ => {
          val xDist = right - left
          left -= xDist
          right += xDist
        })

        binsToCoords = collapse(binsToCoords, nDivisions, xFactor, collapseX = true, collapseY = false)
      } else if (yFactor > 1) {
        0.until(yFactor).foreach(_ => {
          val yDist = top - bottom
          bottom -= yDist
          top += yDist
        })

        binsToCoords = collapse(binsToCoords, nDivisions, yFactor, collapseX = false, collapseY = true)
      }

      binsToCoords += (toBinCoords(x, y, left, right, bottom, top, nDivisions) -> (x, y))
    }

    this
  }

  def merge(that: DownsampleCombiner): DownsampleCombiner = {
    if (that.hasPoints)
      that.binsToCoords.foreach { case (_, (x, y)) => merge(x, y) }
    else
      that.initFrame.foreach { case (x, y) => merge(x, y) }

    this
  }

  def copy(): DownsampleCombiner = {
    val c = new DownsampleCombiner(nDivisions)
    c.hasPoints = hasPoints
    if (c.hasPoints) {
      c.binsToCoords ++= binsToCoords
      c.left = left
      c.right = right
      c.bottom = bottom
      c.top = top
    } else
      c.initFrame ++= initFrame
    c
  }

  def clear() = {
    binsToCoords = Map.empty[(Int, Int), (Double, Double)]
    initFrame = FastIndexedSeq.empty[(Double, Double)]
    hasPoints = false
    left = 0.0
    right = 0.0
    bottom = 0.0
    top = 0.0
  }

  def toRes: IndexedSeq[Row] = binsToCoords.map { case (_, (x, y)) => Row(x, y) }.toFastIndexedSeq
}
