package is.hail.rvd

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types._
import is.hail.expr.types.physical.PInt64
import is.hail.sparkextras._
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.{DefaultFormats, Formats, JValue, ShortTypeHints}
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

  def readLocal(hc: HailContext, path: String, rowType: TStruct, codecSpec: CodecSpec, partFiles: Array[String], requestedType: TStruct): IndexedSeq[Row] = {
    assert(partFiles.length == 1)

    val hConf = hc.hadoopConf
    partFiles.flatMap { p =>
      val f = path + "/parts/" + p
      hConf.readFile(f) { in =>
        using(RVDContext.default) { ctx =>
          HailContext.readRowsPartition(codecSpec.buildDecoder(rowType, requestedType))(ctx, in)
            .map { rv =>
              val r = SafeRow(requestedType.physicalType, rv.region, rv.offset)
              ctx.region.clear()
              r
            }.toFastIndexedSeq
        }
      }
    }
  }
}

abstract class RVDSpec {
  def read(hc: HailContext, path: String, requestedType: TStruct): OrderedRVD

  def readLocal(hc: HailContext, path: String, requestedType: TStruct): IndexedSeq[Row]

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
  def read(hc: HailContext, path: String, requestedType: TStruct): OrderedRVD =
    OrderedRVD.unkeyed(
      requestedType,
      hc.readRows(path, rowType, codecSpec, partFiles, requestedType))

  def readLocal(hc: HailContext, path: String, requestedType: TStruct): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, rowType, codecSpec, partFiles, requestedType)
}

case class OrderedRVDSpec(
  orvdType: OrderedRVDType,
  codecSpec: CodecSpec,
  partFiles: Array[String],
  jRangeBounds: JValue) extends RVDSpec {
  def read(hc: HailContext, path: String, requestedType: TStruct): OrderedRVD = {
    val requestedORVDType = orvdType.copy(rowType = requestedType)
    assert(requestedORVDType.kType == orvdType.kType)

    val rangeBoundsType = TArray(TInterval(requestedORVDType.kType))
    OrderedRVD(requestedORVDType,
      new OrderedRVDPartitioner(requestedORVDType.kType,
        JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]]),
      hc.readRows(path, orvdType.rowType, codecSpec, partFiles, requestedType))
  }

  def readLocal(hc: HailContext, path: String, requestedType: TStruct): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, orvdType.rowType, codecSpec, partFiles, requestedType)
}

case class PersistedRVRDD(
  persistedRDD: RDD[Array[Byte]],
  iterationRDD: ContextRDD[RVDContext, RegionValue])

object RVD {
  def writeLocalUnpartitioned(hc: HailContext, path: String, rowType: TStruct, codecSpec: CodecSpec, rows: IndexedSeq[Annotation]): Array[Long] = {
    val hConf = hc.hadoopConf
    hConf.mkDir(path + "/parts")

    val part0Count =
      hConf.writeFile(path + "/parts/part-0") { os =>
        using(RVDContext.default) { ctx =>
          val rvb = ctx.rvb
          val region = ctx.region
          RichContextRDDRegionValue.writeRowsPartition(codecSpec.buildEncoder(rowType))(ctx,
            rows.iterator.map { a =>
              rvb.start(rowType)
              rvb.addAnnotation(rowType, a)
              RegionValue(region, rvb.end())
            }, os)
        }
      }

    val spec = UnpartitionedRVDSpec(rowType, codecSpec, Array("part-0"))
    spec.write(hConf, path)

    Array(part0Count)
  }

  def union(rvds: Seq[RVD]): RVD = rvds match {
    case Seq(x) => x
    case first +: _ =>
      val sc = first.sparkContext
      OrderedRVD.unkeyed(first.rowType, ContextRDD.union(sc, rvds.map(_.crdd)))
  }

  val memoryCodec = CodecSpec.defaultUncompressed

  val wireCodec = memoryCodec

  def regionValueToBytes(
    makeEnc: OutputStream => Encoder,
    ctx: RVDContext
  )(rv: RegionValue
  ): Array[Byte] =
    using(new ByteArrayOutputStream()) { baos =>
      using(makeEnc(baos)) { enc =>
        enc.writeRegionValue(rv.region, rv.offset)
        enc.flush()
        ctx.region.clear()
        baos.toByteArray
      }
    }

  def bytesToRegionValue(
    makeDec: InputStream => Decoder,
    r: Region,
    carrierRv: RegionValue
  )(bytes: Array[Byte]
  ): RegionValue =
    using(new ByteArrayInputStream(bytes)) { bais =>
      using(makeDec(bais)) { dec =>
        carrierRv.setOffset(dec.readRegionValue(r))
        carrierRv
      }
    }
}

trait RVD {
  self =>

  def rowType: TStruct

  def crdd: ContextRDD[RVDContext, RegionValue]

  private[rvd] def stabilize(
    unstable: ContextRDD[RVDContext, RegionValue],
    codec: CodecSpec = RVD.memoryCodec
  ): ContextRDD[RVDContext, Array[Byte]]

  private[rvd] def destabilize(
    stable: ContextRDD[RVDContext, Array[Byte]],
    codec: CodecSpec = RVD.memoryCodec
  ): ContextRDD[RVDContext, RegionValue]

  private[rvd] def stably(
    f: ContextRDD[RVDContext, Array[Byte]] => ContextRDD[RVDContext, Array[Byte]]
  ): ContextRDD[RVDContext, RegionValue]

  private[rvd] def stably(
    unstable: ContextRDD[RVDContext, RegionValue],
    f: ContextRDD[RVDContext, Array[Byte]] => ContextRDD[RVDContext, Array[Byte]]
  ): ContextRDD[RVDContext, RegionValue]

  private[rvd] def crddBoundary: ContextRDD[RVDContext, RegionValue]

  def boundary: RVD

  def encodedRDD(codec: CodecSpec): RDD[Array[Byte]]

  def head(n: Long, partitionCounts: Option[IndexedSeq[Long]]): RVD

  def collectAsBytes(codec: CodecSpec): Array[Array[Byte]]

  def collect(codec: CodecSpec): Array[Row]

  def forall(p: RegionValue => Boolean): Boolean

  def exists(p: RegionValue => Boolean): Boolean

  def sparkContext: SparkContext

  def getNumPartitions: Int

  def filter(f: (RegionValue) => Boolean): RVD

  def filterWithContext[C](makeContext: (Int, RVDContext) => C, f: (C, RegionValue) => Boolean): RVD

  def mapPartitions(newRowType: TStruct)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): RVD

  def mapPartitions(newRowType: TStruct, f: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]): RVD

  def find(codec: CodecSpec, p: (RegionValue) => Boolean): Option[Array[Byte]]

  def find(region: Region)(p: (RegionValue) => Boolean): Option[RegionValue]

  // Only use on CRDD's whose T is not dependent on the context
  private[rvd] def clearingRun[T: ClassTag](
    crdd: ContextRDD[RVDContext, T]
  ): RDD[T] = crdd.cmap { (ctx, v) =>
    ctx.region.clear()
    v
  }.run

  def map[T](f: (RegionValue) => T)(implicit tct: ClassTag[T]): RDD[T]

  def mapPartitionsWithIndex(newRowType: TStruct, f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]): RVD

  def mapPartitionsWithIndex[T](f: (Int, Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T]

  def mapPartitionsWithIndex[T: ClassTag](
    f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T]

  def mapPartitions[T](f: (Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T]

  def mapPartitions[T: ClassTag](
    f: (RVDContext, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T]

  def treeAggregate[U: ClassTag](zeroValue: U)(
    seqOp: (U, RegionValue) => U,
    combOp: (U, U) => U,
    depth: Int = treeAggDepth(HailContext.get, crdd.getNumPartitions)
  ): U

  def treeAggregateWithPartitionOp[PC, U: ClassTag](zeroValue: U)(
    makePC: (Int, RVDContext) => PC,
    seqOp: (PC, U, RegionValue) => U,
    combOp: (U, U) => U,
    depth: Int = treeAggDepth(HailContext.get, crdd.getNumPartitions)
  ): U

  def aggregateWithPartitionOp[PC, U: ClassTag](
    zeroValue: U, makePC: (Int, RVDContext) => PC
  )(seqOp: (PC, U, RegionValue) => Unit, combOp: (U, U) => U): U

  def aggregate[U: ClassTag](
    zeroValue: U
  )(seqOp: (U, RegionValue) => U,
    combOp: (U, U) => U
  ): U

  def count(): Long

  def countPerPartition(): Array[Long]

  def collectPerPartition[T : ClassTag](f: (Int, RVDContext, Iterator[RegionValue]) => T): Array[T]

  def storageLevel: StorageLevel

  def persist(level: StorageLevel): RVD

  def cache(): RVD

  def unpersist(): RVD

  def coalesce(maxPartitions: Int, shuffle: Boolean): RVD

  def zipWithIndex(name: String, partitionCounts: Option[IndexedSeq[Long]] = None): RVD

  protected def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]): RVDSpec

  def write(path: String, stageLocally: Boolean, codecSpec: CodecSpec): Array[Long]

  def toRows: RDD[Row]

  def subsetPartitions(keep: Array[Int]): RVD

  def cast(newRowType: TStruct): RVD
}
