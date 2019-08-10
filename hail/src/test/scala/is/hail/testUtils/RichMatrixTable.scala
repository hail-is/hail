package is.hail.testUtils

import is.hail.annotations._
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichMatrixTable(vsm: MatrixTable) {
  def rdd: RDD[(Annotation, (Annotation, Iterable[Annotation]))] = {
    val fullRowType = vsm.rvRowPType
    val localEntriesIndex = vsm.entriesIdx
    val localRowType = vsm.rowType
    val rowKeyF = vsm.rowKeysF
    vsm.rvd.map { rv =>
      val fullRow = SafeRow(fullRowType, rv.region, rv.offset)
      val row = fullRow.deleteField(localEntriesIndex)
      (rowKeyF(fullRow), (row, fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)))
    }
  }

  def variantRDD: RDD[(Variant, (Annotation, Iterable[Annotation]))] =
    rdd.map { case (v, (va, gs)) =>
      Variant.fromLocusAlleles(v) -> ((va, gs))
    }

  def typedRDD[RK](implicit rkct: ClassTag[RK]): RDD[(RK, (Annotation, Iterable[Annotation]))] =
    rdd.map { case (v, (va, gs)) =>
      (v.asInstanceOf[RK], (va, gs))
    }

  def variants: RDD[Variant] = variantRDD.keys

  def variantsAndAnnotations: RDD[(Variant, Annotation)] =
    variantRDD.map { case (v, (va, gs)) => (v, va) }
}
