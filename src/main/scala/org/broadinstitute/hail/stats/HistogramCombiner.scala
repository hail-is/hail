package org.broadinstitute.hail.stats

import org.broadinstitute.hail.annotations.Annotation
import java.util.Arrays.binarySearch

import org.broadinstitute.hail.expr._

object HistogramCombiner {
  def schema: Type = TStruct(
    "binEdges" -> TArray(TDouble),
    "binFrequencies" -> TArray(TLong),
    "nSmaller" -> TLong,
    "nGreater" -> TLong)
}

class HistogramCombiner(indices: Array[Double]) extends Serializable {

  val min = indices.head
  val max = indices(indices.length - 1)

  var nSmaller = 0L
  var nGreater = 0L
  val density = Array.fill(indices.length - 1)(0L)

  def merge(d: Double): HistogramCombiner = {
    if (d < min)
      nSmaller += 1
    else if (d > max)
      nGreater += 1
    else {
      val bs = binarySearch(indices, d)
      val ind = if (bs < 0)
        -bs - 2
      else
        math.min(bs, density.length - 1)
      assert(ind < density.length && ind >= 0, s"""found out of bounds index $ind
        |  Resulted from trying to merge $d
        |  Indices are [${indices.mkString(", ")}]
        |  Binary search index was $bs""".stripMargin)
      density(ind) += 1
    }

    this
  }

  def merge(that: HistogramCombiner): HistogramCombiner = {
    require(density.length == that.density.length)

    nSmaller += that.nSmaller
    nGreater += that.nGreater
    for (i <- density.indices)
      density(i) += that.density(i)

    this
  }

  def toAnnotation: Annotation = Annotation(indices: IndexedSeq[Double], density: IndexedSeq[Long], nSmaller, nGreater)
}
