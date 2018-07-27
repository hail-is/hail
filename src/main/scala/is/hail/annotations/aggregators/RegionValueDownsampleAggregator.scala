package is.hail.annotations.aggregators
import is.hail.annotations.RegionValueBuilder
import is.hail.annotations._
import is.hail.expr.types.{TArray, TFloat64, TTuple}
import is.hail.utils.fatal

object RegionValueDownsampleAggregator {
  val typ = TArray(TTuple(Array(TFloat64(), TFloat64())))

  def toBoxes(x: Double, y: Double, xmin: Double, xmax: Double, ymin: Double, ymax: Double, nDivisions: Int): (Int, Int) = {
    var i = (x - xmin) / (xmax - xmin)
    if (i.toInt == 1)
      i *= nDivisions - 1
    else if (i.toInt == 0)
      i *= nDivisions
    else
      fatal(s"""Coordinate x value out of bounds: got x = $x, xmin = $xmin, xmax = $xmax""")

    var j = (y - ymin) / (ymax - ymin)
    if (j.toInt == 1)
      j *= nDivisions - 1
    else if (j.toInt == 0)
      j *= nDivisions
    else
      fatal(s"""Coordinate y value out of bounds: got y = $y, ymin = $ymin, ymax = $ymax""")

    (i.toInt, j.toInt)
  }

  def toCoords(i: Int, j: Int, xmin: Double, xmax: Double, ymin: Double, ymax: Double, nDivisions: Int): (Double, Double) = {
    val x = xmin + ((i + 0.5) / nDivisions * (xmax - xmin))
    val y = ymin + ((j + 0.5) / nDivisions * (ymax - ymin))
    (x, y)
  }
}

class RegionValueDownsampleAggregator(xmin: Double, xmax: Double, ymin: Double, ymax: Double, nDivisions: Int) extends RegionValueAggregator {
  if (xmax <= xmin)
    fatal(s"""xmax value must be greater than xmin value, got xmin = $xmin and xmax = $xmax""")
  if (ymax <= ymin)
    fatal(s"""ymax value must be greater than ymin value, got ymin = $ymin and ymax = $ymax""")
  if (nDivisions < 1)
    fatal(s"""nDivisions value must be greater than or equal to 1, got nDivisions = $nDivisions""")

  private var boxes = Set.empty[(Int, Int)]

  def seqOp(r: Region, offset: Long, missing: Boolean) {
    if (!missing) {
      val row = UnsafeRow.read(TTuple(Array(TFloat64(), TFloat64())), r, offset).asInstanceOf[UnsafeRow]
      val x = row.getDouble(0)
      val y = row.getDouble(1)
      boxes += RegionValueDownsampleAggregator.toBoxes(x, y, xmin, xmax, ymin, ymax, nDivisions)
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    boxes = boxes ++ agg2.asInstanceOf[RegionValueDownsampleAggregator].boxes
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startArray(boxes.size)
    boxes.iterator.foreach { case (i, j) =>
      val (x, y) = RegionValueDownsampleAggregator.toCoords(i, j, xmin, xmax, ymin, ymax, nDivisions)
      rvb.startTuple()
      rvb.addDouble(x)
      rvb.addDouble(y)
      rvb.endTuple()
    }
    rvb.endArray()
  }

  def newInstance(): RegionValueAggregator = new RegionValueDownsampleAggregator(xmin, xmax, ymin, ymax, nDivisions)

  def copy(): RegionValueDownsampleAggregator = {
    val rva = new RegionValueDownsampleAggregator(xmin, xmax, ymin, ymax, nDivisions)
    rva.boxes ++= boxes
    rva
  }

  def clear() {
    boxes = Set.empty[(Int, Int)]
  }
}
