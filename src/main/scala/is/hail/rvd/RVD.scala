package is.hail.rvd

import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.expr.types.Type
import is.hail.utils._
import org.apache.spark.{Partition, SparkContext}
import org.apache.spark.rdd.{AggregateWithContext, RDD}
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

object RVD {
  def apply(rowType: Type, rdd: RDD[RegionValue]): ConcreteRVD = new ConcreteRVD(rowType, rdd)

  def empty(sc: SparkContext, rowType: Type) = ConcreteRVD.empty(sc, rowType)
}

case class PersistedRVRDD(
  persistedRDD: RDD[RegionValue],
  iterationRDD: RDD[RegionValue])

trait RVD {
  self =>
  def rowType: Type

  def rdd: RDD[RegionValue]

  def sparkContext: SparkContext = rdd.sparkContext

  def getNumPartitions: Int = rdd.getNumPartitions

  def partitions: Array[Partition] = rdd.partitions

  def filter(f: (RegionValue) => Boolean): RVD = RVD(rowType, rdd.filter(f))

  def map(newRowType: Type)(f: (RegionValue) => RegionValue): RVD = RVD(newRowType, rdd.map(f))

  def mapWithContext[C](newRowType: Type)(makeContext: () => C)(f: (C, RegionValue) => RegionValue) =
    RVD(newRowType, rdd.mapPartitions { it =>
      val c = makeContext()
      it.map { rv => f(c, rv) }
    })

  def map[T](f: (RegionValue) => T)(implicit tct: ClassTag[T]): RDD[T] = rdd.map(f)

  def mapPartitions(newRowType: Type)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): RVD = RVD(newRowType, rdd.mapPartitions(f))

  def mapPartitionsWithIndex[T](f: (Int, Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T] = rdd.mapPartitionsWithIndex(f)

  def mapPartitions[T](f: (Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T] = rdd.mapPartitions(f)

  def treeAggregate[U: ClassTag](zeroValue: U)(
    seqOp: (U, RegionValue) => U,
    combOp: (U, U) => U,
    depth: Int = 2): U = rdd.treeAggregate(zeroValue)(seqOp, combOp, depth)

  def aggregateWithContext[U: ClassTag, V](context: () => V)(zeroValue: U)
    (seqOp: (V, U, RegionValue) => U, combOp: (U, U) => U): U = {
    AggregateWithContext.aggregateWithContext(rdd)(context)(zeroValue)(seqOp, combOp)
  }

  def count(): Long = rdd.count()

  def countPerPartition(): Array[Long] = rdd.countPerPartition()

  protected def persistRVRDD(level: StorageLevel): PersistedRVRDD = {
    val localRowType = rowType

    // copy, persist region values
    val persistedRDD = rdd.mapPartitions { it =>
      val region = Region()
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

    PersistedRVRDD(persistedRDD,
      persistedRDD
        .mapPartitions { it =>
          val region = Region()
          val rv2 = RegionValue(region)
          it.map { rv =>
            region.setFrom(rv.region)
            rv2.setOffset(rv.offset)
            rv2
          }
        })
  }

  def storageLevel: StorageLevel = StorageLevel.NONE

  def persist(level: StorageLevel): RVD = {
    val PersistedRVRDD(persistedRDD, iterationRDD) = persistRVRDD(level)
    new RVD {
      val rowType: Type = self.rowType

      val rdd: RDD[RegionValue] = iterationRDD

      override def storageLevel: StorageLevel = persistedRDD.getStorageLevel

      override def persist(newLevel: StorageLevel): RVD = {
        if (newLevel == StorageLevel.NONE)
          unpersist()
        else {
          persistedRDD.persist(newLevel)
          this
        }
      }

      override def unpersist(): RVD = {
        persistedRDD.unpersist()
        self
      }
    }
  }

  def cache(): RVD = persist(StorageLevel.MEMORY_ONLY)

  def unpersist(): RVD = this

  def coalesce(maxPartitions: Int, shuffle: Boolean): RVD = RVD(rowType, rdd.coalesce(maxPartitions, shuffle = shuffle))

  def sample(withReplacement: Boolean, p: Double, seed: Long): RVD =
    RVD(rowType, rdd.sample(withReplacement, p, seed))
}
