package is.hail.rvd

import is.hail.annotations.RegionValue
import is.hail.utils._
import is.hail.expr.types.TStruct
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object UnpartitionedRVD {
  def empty(sc: SparkContext, rowType: TStruct): UnpartitionedRVD = new UnpartitionedRVD(rowType, sc.emptyRDD[RegionValue])
}

class UnpartitionedRVD(val rowType: TStruct, val rdd: RDD[RegionValue]) extends RVD {
  self =>

  def filter(f: (RegionValue) => Boolean): UnpartitionedRVD = new UnpartitionedRVD(rowType, rdd.filter(f))

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

  def sample(withReplacement: Boolean, p: Double, seed: Long): UnpartitionedRVD =
    new UnpartitionedRVD(rowType, rdd.sample(withReplacement, p, seed))

  def write(path: String): (RVDSpec, Array[Long]) = {
    val (partFiles, partitionCounts) = rdd.writeRows(path, rowType)
    (UnpartitionedRVDSpec(rowType, partFiles),
      partitionCounts)
  }

  def coalesce(maxPartitions: Int, shuffle: Boolean): UnpartitionedRVD = new UnpartitionedRVD(rowType, rdd.coalesce(maxPartitions, shuffle = shuffle))
}
