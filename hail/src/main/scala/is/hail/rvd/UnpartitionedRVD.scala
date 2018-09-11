package is.hail.rvd

import is.hail.annotations.{KeyedRow, RegionValue, UnsafeRow}
import is.hail.expr.types.{ TInt64, TStruct }
import is.hail.sparkextras._
import is.hail.io.CodecSpec
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object UnpartitionedRVD {
  def empty(sc: SparkContext, rowType: TStruct): RVD =
    UnpartitionedRVD(rowType, ContextRDD.empty[RVDContext, RegionValue](sc))

  def apply(rowType: TStruct, crdd: ContextRDD[RVDContext, RegionValue]): RVD = {
    new UnpartitionedRVD(rowType, crdd)
  }
}

class UnpartitionedRVD private (val rowType: TStruct, val crdd: ContextRDD[RVDContext, RegionValue]) extends RVD {
  self =>

  override def toOrderedRVD: OrderedRVD = {
    new OrderedRVD(
      typ = OrderedRVDType(FastIndexedSeq.empty, rowType),
      partitioner = OrderedRVDPartitioner.unkeyed(crdd.getNumPartitions),
      crdd = crdd
    )
  }

  def boundary = new UnpartitionedRVD(rowType, crddBoundary)

  def head(n: Long, partitionCounts: Option[IndexedSeq[Long]]): UnpartitionedRVD =
    new UnpartitionedRVD(rowType, crdd.head(n, partitionCounts))

  def filter(f: (RegionValue) => Boolean): UnpartitionedRVD =
    new UnpartitionedRVD(rowType, crddBoundary.filter(f))

  def filterWithContext[C](makeContext: (Int, RVDContext) => C, f: (C, RegionValue) => Boolean): RVD = {
    mapPartitionsWithIndex(rowType, { (i, context, it) =>
      val c = makeContext(i, context)
      it.filter { rv =>
        if (f(c, rv))
          true
        else {
          rv.region.clear()
          false
        }
      }
    })
  }

  def persist(level: StorageLevel): UnpartitionedRVD = {
    val PersistedRVRDD(persistedRDD, iterationRDD) = persistRVRDD(level)
    new UnpartitionedRVD(rowType, iterationRDD) {
      override def storageLevel: StorageLevel = persistedRDD.getStorageLevel

      override def persist(newLevel: StorageLevel): UnpartitionedRVD = {
        if (newLevel == StorageLevel.NONE)
          unpersist()
        else {
          persistedRDD.persist(newLevel)
          this
        }
      }

      override def unpersist(): UnpartitionedRVD = {
        persistedRDD.unpersist()
        self
      }
    }
  }

  def zipWithIndex(name: String, partitionCounts: Option[IndexedSeq[Long]] = None): UnpartitionedRVD = {
    val (newRowType, newCRDD) = zipWithIndexCRDD(name, partitionCounts)

    new UnpartitionedRVD(
      rowType = newRowType,
      crdd = newCRDD
    )
  }

  override protected def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]): RVDSpec =
    UnpartitionedRVDSpec(rowType, codecSpec, partFiles)

  def coalesce(maxPartitions: Int, shuffle: Boolean): UnpartitionedRVD =
    new UnpartitionedRVD(
      rowType,
      if (shuffle)
        stably(_.shuffleCoalesce(maxPartitions))
      else
        crdd.noShuffleCoalesce(maxPartitions))

  def subsetPartitions(keep: Array[Int]): UnpartitionedRVD = {
    require(keep.length <= crdd.partitions.length, "tried to subset to more partitions than exist")
    require(keep.isIncreasing && (keep.isEmpty || (keep.head >= 0 && keep.last < crdd.partitions.length)),
      "values not increasing or not in range [0, number of partitions)")
    
    new UnpartitionedRVD(rowType, crdd.subsetPartitions(keep))
  }

  override def toUnpartitionedRVD: RVD = this
}
