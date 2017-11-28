package is.hail.stats

import java.util.Arrays.binarySearch

import is.hail.annotations.Annotation
import is.hail.expr._

object HistogramCombiner {
  def schema: Type = TStruct(
    "binEdges" -> TArray(TFloat64()),
    "binFrequencies" -> TArray(TInt64()),
    "nLess" -> TInt64(),
    "nGreater" -> TInt64())
}

class HistogramCombiner(indices: Array[Double]) extends Serializable {

  val min = indices.head
  val max = indices(indices.length - 1)

  var nLess = 0L
  var nGreater = 0L
  val frequency = Array.fill(indices.length - 1)(0L)

  def merge(d: Double): HistogramCombiner = {
    if (d < min)
      nLess += 1
    else if (d > max)
      nGreater += 1
    else if (!d.isNaN) {
      val bs = binarySearch(indices, d)
      val ind = if (bs < 0)
        -bs - 2
      else
        math.min(bs, frequency.length - 1)
      assert(ind < frequency.length && ind >= 0,
        s"""found out of bounds index $ind
            |  Resulted from trying to merge $d
            |  Indices are [${ indices.mkString(", ") }]
            |  Binary search index was $bs""".stripMargin)
      frequency(ind) += 1
    }

    this
  }

  def merge(that: HistogramCombiner): HistogramCombiner = {
    require(frequency.length == that.frequency.length)

    nLess += that.nLess
    nGreater += that.nGreater
    for (i <- frequency.indices)
      frequency(i) += that.frequency(i)

    this
  }

  def toAnnotation: Annotation = Annotation(indices: IndexedSeq[Double], frequency: IndexedSeq[Long], nLess, nGreater)
}
