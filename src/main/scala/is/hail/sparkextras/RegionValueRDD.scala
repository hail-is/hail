package is.hail.sparkextras

import is.hail.annotations.{MemoryBuffer, RegionValue, RegionValueBuilder}
import is.hail.expr.TStruct
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class RegionValueRDD(val rdd: RDD[RegionValue], val rowType: TStruct) extends Serializable {
  def getStorageLevel2: StorageLevel = StorageLevel.NONE

  def unpersist2(): RegionValueRDD = throw new IllegalArgumentException("not persisted")

  def persist2(level: StorageLevel): RegionValueRDD = {
    val localRowType = rowType

    // copy, persist region values
    val persistedRDD = rdd.mapPartitions { it =>
      val region = MemoryBuffer()
      val rvb = new RegionValueBuilder(region)
      it.map { rv =>
        region.clear()
        rvb.start(localRowType)
        rvb.addRegionValue(localRowType, rv)
        val off = rvb.end()
        RegionValue(region.copy(), off)
      }
    }
      .persist(level)

    val rdd2 = persistedRDD
      .mapPartitions { it =>
        val region = MemoryBuffer()
        val rv2 = RegionValue(region)
        it.map { rv =>
          region.setFrom(rv.region)
          rv2.setOffset(rv.offset)
          rv2
        }
      }

    val self = this

    new RegionValueRDD(rdd2, rowType) {
      override def getStorageLevel2: StorageLevel = persistedRDD.getStorageLevel

      override def persist2(newLevel: StorageLevel): RegionValueRDD = {
        if (newLevel == getStorageLevel2)
          this
        else
          self.persist2(newLevel)
      }

      override def unpersist2(): RegionValueRDD = {
        persistedRDD.unpersist()
        self
      }
    }
  }
}
