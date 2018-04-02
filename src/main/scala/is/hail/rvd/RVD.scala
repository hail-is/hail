package is.hail.rvd

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{JSONAnnotationImpex, Parser}
import is.hail.expr.types.{TArray, TInterval, TStruct, TStructSerializer}
import is.hail.sparkextras._
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop
import org.apache.spark.{Partition, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.{CustomSerializer, DefaultFormats, Formats, JValue, ShortTypeHints}
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.reflect.ClassTag

object RVDSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[RVDSpec], classOf[UnpartitionedRVDSpec], classOf[OrderedRVDSpec],
      classOf[CodecSpec], classOf[PackCodecSpec], classOf[BlockBufferSpec],
      classOf[LZ4BlockBufferSpec], classOf[StreamBlockBufferSpec],
      classOf[BufferSpec], classOf[LEB128BufferSpec], classOf[BlockingBufferSpec]))
    override val typeHintFieldName = "name"
  } +
    new TStructSerializer +
    new OrderedRVDTypeSerializer

  def read(hc: HailContext, path: String): RVDSpec = {
    val metadataFile = path + "/metadata.json.gz"
    val jv = hc.hadoopConf.readFile(metadataFile) { in => JsonMethods.parse(in) }
    jv.extract[RVDSpec]
  }

  def readLocal(hc: HailContext, path: String, rowType: TStruct, codecSpec: CodecSpec, partFiles: Array[String]): IndexedSeq[Row] = {
    assert(partFiles.length == 1)

    val hConf = hc.hadoopConf
    partFiles.flatMap { p =>
      val f = path + "/parts/" + p
      val in = hConf.unsafeReader(f)
      using(RVDContext.default) { ctx =>
        HailContext.readRowsPartition(rowType, codecSpec)(ctx, in)
          .map(rv => Annotation.safeFromBaseStructRegionValue(rowType, rv.region, rv.offset))
      }
    }.toFastIndexedSeq
  }
}

abstract class RVDSpec {
  def read(hc: HailContext, path: String): RVD

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row]

  def write(hadoopConf: hadoop.conf.Configuration, path: String) {
    hadoopConf.writeTextFile(path + "/metadata.json.gz") { out =>
      implicit val formats = RVDSpec.formats
      Serialization.write(this, out)
    }
  }
}

case class UnpartitionedRVDSpec(
  rowType: TStruct,
  codecSpec: CodecSpec,
  partFiles: Array[String]) extends RVDSpec {
  def read(hc: HailContext, path: String): UnpartitionedRVD =
    new UnpartitionedRVD(
      rowType,
      hc.readRows(path, rowType, codecSpec, partFiles))

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, rowType, codecSpec, partFiles)
}

case class OrderedRVDSpec(
  orvdType: OrderedRVDType,
  codecSpec: CodecSpec,
  partFiles: Array[String],
  jRangeBounds: JValue) extends RVDSpec {
  def read(hc: HailContext, path: String): OrderedRVD = {
    val rangeBoundsType = TArray(TInterval(orvdType.pkType))
    OrderedRVD(orvdType,
      new OrderedRVDPartitioner(orvdType.partitionKey, orvdType.kType,
          JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType).asInstanceOf[IndexedSeq[Interval]]),
      hc.readRows(path, orvdType.rowType, codecSpec, partFiles))
  }

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, orvdType.rowType, codecSpec, partFiles)
}

case class PersistedRVRDD(
  persistedRDD: RDD[Array[Byte]],
  iterationRDD: ContextRDD[RVDContext, RegionValue])

object RVD {
  def writeLocalUnpartitioned(hc: HailContext, path: String, rowType: TStruct, codecSpec: CodecSpec, rows: IndexedSeq[Annotation]): Array[Long] = {
    val hConf = hc.hadoopConf
    hConf.mkDir(path + "/parts")

    val os = hConf.unsafeWriter(path + "/parts/part-0")
    val part0Count = using(Region()) { region =>
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)
      RichContextRDDRegionValue.writeRowsPartition(rowType, codecSpec)(0,
        rows.iterator.map { a =>
          region.clear()
          rvb.start(rowType)
          rvb.addAnnotation(rowType, a)
          rv.setOffset(rvb.end())
          rv
        }, os)
    }

    val spec = UnpartitionedRVDSpec(rowType, codecSpec, Array("part-0"))
    spec.write(hConf, path)

    Array(part0Count)
  }
}

trait RVD {
  self =>
  def rowType: TStruct

  def crdd: ContextRDD[RVDContext, RegionValue]

  def rdd: RDD[RegionValue]

  def sparkContext: SparkContext = crdd.sparkContext

  def getNumPartitions: Int = crdd.getNumPartitions

  def partitions: Array[Partition] = crdd.partitions

  def filter(f: (RegionValue) => Boolean): RVD

  def map(newRowType: TStruct)(f: (RegionValue) => RegionValue): UnpartitionedRVD =
    new UnpartitionedRVD(newRowType, crdd.map(f))

  def mapWithContext[C](newRowType: TStruct)(makeContext: () => C)(f: (C, RegionValue) => RegionValue): UnpartitionedRVD =
    new UnpartitionedRVD(newRowType, crdd.mapPartitions { it =>
      val c = makeContext()
      it.map { rv => f(c, rv) }
    })

  // FIXME: just make user call run
  def map[T](f: (RegionValue) => T)(implicit tct: ClassTag[T]): RDD[T] = crdd.map(f).run

  def mapPartitions(newRowType: TStruct)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): RVD =
    new UnpartitionedRVD(newRowType, crdd.mapPartitions(f))

  def mapPartitions(newRowType: TStruct, f: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]): RVD =
    new UnpartitionedRVD(newRowType, crdd.cmapPartitions(f))

  // FIXME: just make user call run
  def mapPartitionsWithIndex[T](f: (Int, Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T] = crdd.mapPartitionsWithIndex(f).run

  // FIXME: just make user call run
  def mapPartitions[T](f: (Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T] = crdd.mapPartitions(f).run

  def constrainToOrderedPartitioner(
    ordType: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner
  ): OrderedRVD

  def treeAggregate[U: ClassTag](zeroValue: U)(
    seqOp: (U, RegionValue) => U,
    combOp: (U, U) => U,
    depth: Int = 2): U =
    // FIXME: 1) need real copyable zeroValue
    crdd.treeAggregate(() => zeroValue, seqOp, combOp, depth)

  def count(): Long = rdd.count()

  def countPerPartition(): Array[Long] = rdd.countPerPartition()

  protected def persistRVRDD(level: StorageLevel): PersistedRVRDD = {
    val localRowType = rowType

    val persistCodec = CodecSpec.default

    // copy, persist region values
    val persistedRDD = crdd.cmapPartitions { (ctx, it) =>
      val baos = new ByteArrayOutputStream()
      val enc = persistCodec.buildEncoder(baos)
      it.map { rv =>
        baos.reset()
        enc.writeRegionValue(localRowType, rv.region, rv.offset)
        enc.flush()
        baos.toByteArray
      }
    }.run.persist(level)

    PersistedRVRDD(persistedRDD,
      ContextRDD.weaken(persistedRDD)
        .cmapPartitions { (ctx, it) =>
          val region = ctx.region
          val rv2 = RegionValue(region)
          it.map { bytes =>
            region.clear()
            val bais = new ByteArrayInputStream(bytes)
            val dec = persistCodec.buildDecoder(bais)
            rv2.setOffset(dec.readRegionValue(localRowType, region))
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

  protected def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]): RVDSpec

  final def write(path: String, codecSpec: CodecSpec): Array[Long] = {
    val (partFiles, partitionCounts) = crdd.writeRows(path, rowType, codecSpec)
    rvdSpec(codecSpec, partFiles).write(sparkContext.hadoopConfiguration, path)
    partitionCounts
  }

  def toRows: RDD[Row] = {
    val localRowType = rowType
    crdd.map { rv => new UnsafeRow(localRowType, rv.region.copy(), rv.offset) }.run
  }
}
