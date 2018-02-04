package is.hail.rvd

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{JSONAnnotationImpex, Parser}
import is.hail.expr.types.{TArray, TStruct}
import is.hail.io.RichRDDRegionValue
import is.hail.utils._
import org.apache.spark.{Partition, SparkContext}
import org.apache.spark.rdd.{AggregateWithContext, RDD}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.JValue
import org.json4s.JsonAST.{JArray, JObject, JString}

import scala.reflect.ClassTag

object RVDSpec {
  def extract(jv: JValue): RVDSpec = {
    jv \ "name" match {
      case JString("UnpartitionedRVDSpec") =>
        UnpartitionedRVDSpec.extract(jv \ "args")
      case JString("OrderedRVDSpec") =>
        OrderedRVDSpec.extract(jv \ "args")
    }
  }

  def readLocal(hc: HailContext, path: String, rowType: TStruct, partFiles: Array[String]): IndexedSeq[Row] = {
    assert(partFiles.length == 1)

    val hConf = hc.hadoopConf
    partFiles.flatMap { p =>
      val f = path + "/parts/" + p
      val in = hConf.unsafeReader(f)
      HailContext.readRowsPartition(rowType)(0, in)
        .map(rv => new UnsafeRow(rowType, rv.region.copy(), rv.offset))
    }
      .toFastIndexedSeq
  }
}

abstract class RVDSpec {
  def read(hc: HailContext, path: String): RVD

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row]

  def toJSON: JValue
}

object UnpartitionedRVDSpec {

  case class JSONArgs(row_type: String, part_files: Array[String])

  def extract(jargs: JValue): UnpartitionedRVDSpec = {
    val args = jargs.extract[JSONArgs]
    UnpartitionedRVDSpec(Parser.parseStructType(args.row_type), args.part_files)
  }
}

case class UnpartitionedRVDSpec(
  rowType: TStruct,
  partFiles: Array[String]) extends RVDSpec {
  def read(hc: HailContext, path: String): UnpartitionedRVD =
    new UnpartitionedRVD(rowType, hc.readRows(path, rowType, partFiles))

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, rowType, partFiles)

  def toJSON: JValue = JObject("name" -> JString("UnpartitionedRVDSpec"),
    "args" -> JObject("row_type" -> JString(rowType.toString),
      "part_files" -> JArray(partFiles.map(JString).toList)))
}

object OrderedRVDSpec {

  case class JSONArgs(ordered_row_type: String, part_files: Array[String], partition_bounds: JValue)

  def extract(jv: JValue): OrderedRVDSpec = {
    val args = jv.extract[JSONArgs]
    val orvdType: OrderedRVDType = Parser.parseOrderedRVDType(args.ordered_row_type)
    val rangeBoundsType = TArray(orvdType.pkType)
    val partitionBounds = UnsafeIndexedSeq(rangeBoundsType,
      JSONAnnotationImpex.importAnnotation(args.partition_bounds, rangeBoundsType).asInstanceOf[IndexedSeq[Annotation]])

    OrderedRVDSpec(orvdType, args.part_files, partitionBounds)
  }
}

case class OrderedRVDSpec(
  orvdType: OrderedRVDType,
  partFiles: Array[String],
  partitionBounds: UnsafeIndexedSeq) extends RVDSpec {
  def read(hc: HailContext, path: String): OrderedRVD =
    OrderedRVD(orvdType,
      new OrderedRVDPartitioner(partFiles.length, orvdType.partitionKey, orvdType.kType, partitionBounds),
      hc.readRows(path, orvdType.rowType, partFiles))

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, orvdType.rowType, partFiles)

  def toJSON: JValue = {
    val rangeBoundsType = TArray(orvdType.pkType)
    JObject("name" -> JString("OrderedRVDSpec"),
      "args" -> JObject("ordered_row_type" -> JString(orvdType.toString),
        "part_files" -> JArray(partFiles.map(JString).toList),
        "partition_bounds" -> JSONAnnotationImpex.exportAnnotation(partitionBounds, rangeBoundsType)))
  }
}

case class PersistedRVRDD(
  persistedRDD: RDD[RegionValue],
  iterationRDD: RDD[RegionValue])

object RVD {
  def writeLocalUnpartitioned(hc: HailContext, path: String, rowType: TStruct, rows: IndexedSeq[Annotation]): (RVDSpec, Array[Long]) = {
    val hConf = hc.hadoopConf
    hConf.mkDir(path + "/parts")

    val os = hConf.unsafeWriter(path + "/parts/part-0")
    val rvb = new RegionValueBuilder()
    val part0Count = RichRDDRegionValue.writeRowsPartition(rowType)(0,
      rows.map { a =>
        val region = Region()
        rvb.set(region)
        rvb.start(rowType)
        rvb.addAnnotation(rowType, a)
        RegionValue(region, rvb.end())
      }.iterator, os)

    val rvdSpec = UnpartitionedRVDSpec(rowType, Array("part-0"))
    val partitionCounts = Array(part0Count)

    (rvdSpec, partitionCounts)
  }
}

trait RVD {
  self =>
  // FIXME TStruct
  def rowType: TStruct

  def rdd: RDD[RegionValue]

  def sparkContext: SparkContext = rdd.sparkContext

  def getNumPartitions: Int = rdd.getNumPartitions

  def partitions: Array[Partition] = rdd.partitions

  def filter(f: (RegionValue) => Boolean): RVD

  def map(newRowType: TStruct)(f: (RegionValue) => RegionValue): UnpartitionedRVD = new UnpartitionedRVD(newRowType, rdd.map(f))

  def mapWithContext[C](newRowType: TStruct)(makeContext: () => C)(f: (C, RegionValue) => RegionValue): UnpartitionedRVD =
    new UnpartitionedRVD(newRowType, rdd.mapPartitions { it =>
      val c = makeContext()
      it.map { rv => f(c, rv) }
    })

  def map[T](f: (RegionValue) => T)(implicit tct: ClassTag[T]): RDD[T] = rdd.map(f)

  def mapPartitions(newRowType: TStruct)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): RVD = new UnpartitionedRVD(newRowType, rdd.mapPartitions(f))

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

  def persist(level: StorageLevel): RVD

  def cache(): RVD = persist(StorageLevel.MEMORY_ONLY)

  def unpersist(): RVD = this

  def coalesce(maxPartitions: Int, shuffle: Boolean): RVD

  def sample(withReplacement: Boolean, p: Double, seed: Long): RVD

  def write(path: String): (RVDSpec, Array[Long])
}
