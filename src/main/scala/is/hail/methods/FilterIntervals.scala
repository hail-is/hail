package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.utils.{ArrayBuilder, Interval, IntervalTree}
import is.hail.variant.{Locus, MatrixTable, Variant}

import scala.collection.JavaConverters._

object FilterIntervals {
  def apply(vsm: MatrixTable, intervals: java.util.ArrayList[Interval[Locus]], keep: Boolean): MatrixTable = {
    implicit val locusOrd = vsm.genomeReference.locusOrdering
    val iList = IntervalTree[Locus](intervals.asScala.toArray)
    apply(vsm, iList, keep)
  }

  def apply[T, U](vsm: MatrixTable, iList: IntervalTree[Locus, U], keep: Boolean): MatrixTable = {
    implicit val locusOrd = vsm.matrixType.locusType.ordering.toOrdering

    val ab = new ArrayBuilder[(Interval[Annotation], Annotation)]()
    iList.foreach { case (i, v) =>
      ab += (Interval[Annotation](i.start, i.end), v)
    }

    val iList2 = IntervalTree.annotationTree(ab.result())

    if (keep)
      vsm.copy(rdd = vsm.rdd.filterIntervals(iList2))
    else {
      val iListBc = vsm.sparkContext.broadcast(iList)
      vsm.filterVariants { (v, va, gs) => !iListBc.value.contains(v.asInstanceOf[Variant].locus) }
    }
  }
}
