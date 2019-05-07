package is.hail.stats

import java.util
import java.util.Arrays.binarySearch

import is.hail.annotations.Annotation
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt64, TStruct}
import is.hail.utils.fatal

object HistogramCombiner {
  def schema: TStruct = TStruct(
    "bin_edges" -> TArray(TFloat64()),
    "bin_freq" -> TArray(TInt64()),
    "n_smaller" -> TInt64(),
    "n_larger" -> TInt64())
}

class HistogramCombiner(val indices: Array[Double]) extends Serializable {
  if (indices.exists(x => x.isInfinite || x.isNaN))
    fatal(s"cannot compute a histogram with NaN or infinite indices:\n  [ ${ indices.mkString(", ") } ]")

  val min = indices.head
  val max = indices(indices.length - 1)

  var nSmaller = 0L
  var nLarger = 0L
  var frequency = Array.fill(indices.length - 1)(0L)

  def merge(d: Double): HistogramCombiner = {
    if (!java.lang.Double.isNaN(d)) {
      binarySearch(indices, d) match {
        case -1 => nSmaller += 1
        case x if (x < 0 && -x - 1 == indices.length) => nLarger += 1
        case x if (x < 0) => frequency(-x - 2) += 1
        case x => frequency(math.min(x, frequency.length - 1)) += 1
      }
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
