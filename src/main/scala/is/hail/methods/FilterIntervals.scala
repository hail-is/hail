package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.utils.{ArrayBuilder, Interval, IntervalTree}
import is.hail.variant.{Locus, MatrixTable, Variant}

import scala.collection.JavaConverters._

object FilterIntervals {
  def apply(vsm: MatrixTable, intervals: java.util.ArrayList[Interval], keep: Boolean): MatrixTable = {
    val pord = vsm.genomeReference.locusType.ordering
    val iList = IntervalTree(vsm.genomeReference.locusType.ordering, intervals.asScala.toArray)
    apply(vsm, iList, keep)
  }

  def apply[U](vsm: MatrixTable, iList: IntervalTree[U], keep: Boolean): MatrixTable = {
    val pord = vsm.genomeReference.locusType.ordering

    val ab = new ArrayBuilder[(Interval, Annotation)]()
    iList.foreach { case (i, v) =>
      ab += (Interval(i.start, i.end), v)
    }

    val iList2 = IntervalTree.annotationTree(pord, ab.result())

    if (keep)
      vsm.copy(rdd = vsm.rdd.filterIntervals(pord, iList2))
    else {
      val iListBc = vsm.sparkContext.broadcast(iList)
      vsm.filterVariants { (v, va, gs) => !iListBc.value.contains(pord, v.asInstanceOf[Variant].locus) }
    }
  }
}
