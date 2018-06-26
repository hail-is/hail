package is.hail.stats

import java.util
import java.util.Arrays.binarySearch

import is.hail.annotations.Annotation
import is.hail.expr.types._

object HistogramCombiner {
  def schema: TStruct = TStruct(
    "bin_edges" -> TArray(TFloat64()),
    "bin_freq" -> TArray(TInt64()),
    "n_smaller" -> TInt64(),
    "n_larger" -> TInt64())
}

class HistogramCombiner(val indices: Array[Double]) extends Serializable {

  val min = indices.head
  val max = indices(indices.length - 1)

  var nSmaller = 0L
  var nLarger = 0L
  var frequency = Array.fill(indices.length - 1)(0L)

  def merge(d: Double): HistogramCombiner = {
    if (d < min)
      nSmaller += 1
    else if (d > max)
      nLarger += 1
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

    nSmaller += that.nSmaller
    nLarger += that.nLarger
    for (i <- frequency.indices)
      frequency(i) += that.frequency(i)

    this
  }

  def toAnnotation: Annotation = Annotation(indices: IndexedSeq[Double], frequency: IndexedSeq[Long], nSmaller, nLarger)

  def clear() {
    nSmaller = 0L
    nLarger = 0L
    util.Arrays.fill(frequency, 0L)
  }

  def copy(): HistogramCombiner = {
    val c = new HistogramCombiner(indices)
    c.nSmaller = nSmaller
    c.nLarger = nLarger
    c.frequency = frequency.clone()
    c
  }
}
