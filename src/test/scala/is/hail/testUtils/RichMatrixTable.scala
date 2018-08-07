package is.hail.testUtils

import is.hail.annotations._
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichMatrixTable(vsm: MatrixTable) {
  def rdd: RDD[(Annotation, (Annotation, Iterable[Annotation]))] = {
    val fullRowType = vsm.rvRowType
    val localEntriesIndex = vsm.entriesIndex
    val localRowType = vsm.rowType
    val rowKeyF = vsm.rowKeysF
    vsm.rvd.map { rv =>
      val unsafeFullRow = new UnsafeRow(fullRowType, rv)
      val fullRow = SafeRow(fullRowType, rv.region, rv.offset)
      val row = fullRow.deleteField(localEntriesIndex)
      (rowKeyF(fullRow), (row, fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)))
    }
  }

  def variantRDD: RDD[(Variant, (Annotation, Iterable[Annotation]))] =
    rdd.map { case (v, (va, gs)) =>
      Variant.fromLocusAlleles(v) -> (va, gs)
    }

  def typedRDD[RK](implicit rkct: ClassTag[RK]): RDD[(RK, (Annotation, Iterable[Annotation]))] =
    rdd.map { case (v, (va, gs)) =>
      (v.asInstanceOf[RK], (va, gs))
    }

  def variants: RDD[Variant] = variantRDD.keys

  def variantsAndAnnotations: RDD[(Variant, Annotation)] =
    variantRDD.map { case (v, (va, gs)) => (v, va) }

  def reorderCols(newIds: Array[Annotation]): MatrixTable = {
    require(newIds.length == vsm.numCols)
    require(newIds.areDistinct())

    val sampleSet = vsm.colKeys.toSet[Annotation]
    val newSampleSet = newIds.toSet

    val notInDataset = newSampleSet -- sampleSet
    if (notInDataset.nonEmpty)
      fatal(s"Found ${ notInDataset.size } ${ plural(notInDataset.size, "sample ID") } in new ordering that are not in dataset:\n  " +
        s"@1", notInDataset.truncatable("\n  "))

    val oldIndex = vsm.colKeys.zipWithIndex.toMap
    val newToOld = newIds.map(oldIndex)

    vsm.chooseCols(newToOld)
  }
}
