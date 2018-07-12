package is.hail.rvd

import java.io.{ ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream }

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types._
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
              val r = SafeRow(requestedType, rv.region, rv.offset)
              ctx.region.clear()
              r
            }.toFastIndexedSeq
        }
      }
    }
  }
}

abstract class RVDSpec {
  def read(hc: HailContext, path: String, requestedType: TStruct): RVD

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
  def read(hc: HailContext, path: String, requestedType: TStruct): UnpartitionedRVD =
    new UnpartitionedRVD(
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
    assert(requestedORVDType.pkType == orvdType.pkType)
    assert(requestedORVDType.kType == orvdType.kType)
    assert(requestedORVDType.partitionKey sameElements orvdType.partitionKey)

    val rangeBoundsType = TArray(TInterval(requestedORVDType.pkType))
    OrderedRVD(requestedORVDType,
      new OrderedRVDPartitioner(requestedORVDType.partitionKey, requestedORVDType.kType,
        JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType).asInstanceOf[IndexedSeq[Interval]]),
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

  def union(rvds: Seq[RVD]): RVD = {
    require(rvds.length > 1)
    val first = rvds.head
    val sc = first.sparkContext
    new UnpartitionedRVD(first.rowType,
      ContextRDD.union(sc, rvds.map(_.crdd)))
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
  ): ContextRDD[RVDContext, Array[Byte]] = {
    val enc = codec.buildEncoder(rowType)
    unstable.cmapPartitions { (ctx, it) =>
      it.map(RVD.regionValueToBytes(enc, ctx))
    }
  }

  private[rvd] def destabilize(
    stable: ContextRDD[RVDContext, Array[Byte]],
    codec: CodecSpec = RVD.memoryCodec
  ): ContextRDD[RVDContext, RegionValue] = {
    val dec = codec.buildDecoder(rowType, rowType)
    stable.cmapPartitions { (ctx, it) =>
      val rv = RegionValue(ctx.region)
      it.map(RVD.bytesToRegionValue(dec, ctx.region, rv))
    }
  }

  private[rvd] def stably(
    f: ContextRDD[RVDContext, Array[Byte]] => ContextRDD[RVDContext, Array[Byte]]
  ): ContextRDD[RVDContext, RegionValue] = stably(crdd, f)

  private[rvd] def stably(
    unstable: ContextRDD[RVDContext, RegionValue],
    f: ContextRDD[RVDContext, Array[Byte]] => ContextRDD[RVDContext, Array[Byte]]
  ): ContextRDD[RVDContext, RegionValue] = destabilize(f(stabilize(unstable)))

  private[rvd] def crddBoundary: ContextRDD[RVDContext, RegionValue] =
    crdd.cmapPartitionsAndContext { (consumerCtx, part) =>
      val producerCtx = consumerCtx.freshContext
      val it = part.flatMap(_ (producerCtx))
      new Iterator[RegionValue]() {
        private[this] var cleared: Boolean = false

        def hasNext = {
          if (!cleared) {
            cleared = true
            producerCtx.region.clear()
          }
          it.hasNext
        }

        def next = {
          if (!cleared) {
            producerCtx.region.clear()
          }
          cleared = false
          it.next
        }
      }
    }

  def boundary: RVD

  def encodedRDD(codec: CodecSpec): RDD[Array[Byte]] =
    stabilize(crdd, codec).run

  def head(n: Long): RVD

  final def takeAsBytes(n: Int, codec: CodecSpec): Array[Array[Byte]] =
    head(n).encodedRDD(codec).collect()

  final def take(n: Int, codec: CodecSpec): Array[Row] = {
    val dec = codec.buildDecoder(rowType, rowType)
    val encodedData = takeAsBytes(n, codec)
    Region.scoped { region =>
      encodedData.iterator
        .map(RVD.bytesToRegionValue(dec, region, RegionValue(region)))
        .map { rv =>
          val row = SafeRow(rowType, rv)
          region.clear()
          row
        }.toArray
    }
  }

  def forall(p: RegionValue => Boolean): Boolean =
    crdd.map(p).run.forall(x => x)

  def exists(p: RegionValue => Boolean): Boolean =
    crdd.map(p).run.exists(x => x)

  def sparkContext: SparkContext = crdd.sparkContext

  def getNumPartitions: Int = crdd.getNumPartitions

  def filter(f: (RegionValue) => Boolean): RVD

  def map(newRowType: TStruct)(f: (RegionValue) => RegionValue): UnpartitionedRVD =
    new UnpartitionedRVD(newRowType, crdd.map(f))

  def mapWithContext[C](newRowType: TStruct)(makeContext: () => C)(f: (C, RegionValue) => RegionValue): UnpartitionedRVD =
    new UnpartitionedRVD(newRowType, crdd.mapPartitions { it =>
      val c = makeContext()
      it.map { rv => f(c, rv) }
    })

  def mapPartitions(newRowType: TStruct)(f: (Iterator[RegionValue]) => Iterator[RegionValue]): RVD =
    new UnpartitionedRVD(newRowType, crdd.mapPartitions(f))

  def mapPartitions(newRowType: TStruct, f: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]): RVD =
    new UnpartitionedRVD(newRowType, crdd.cmapPartitions(f))

  def find(codec: CodecSpec, p: (RegionValue) => Boolean): Option[Array[Byte]] =
    filter(p).takeAsBytes(1, codec).headOption

  def find(region: Region)(p: (RegionValue) => Boolean): Option[RegionValue] =
    find(RVD.wireCodec, p).map(
      RVD.bytesToRegionValue(RVD.wireCodec.buildDecoder(rowType, rowType), region, RegionValue(region)))

  // Only use on CRDD's whose T is not dependent on the context
  private[rvd] def clearingRun[T: ClassTag](
    crdd: ContextRDD[RVDContext, T]
  ): RDD[T] = crdd.cmap { (ctx, v) =>
    ctx.region.clear()
    v
  }.run

  def map[T](f: (RegionValue) => T)(implicit tct: ClassTag[T]): RDD[T] = clearingRun(crdd.map(f))

  def mapPartitionsWithIndex(newRowType: TStruct, f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[RegionValue]): RVD =
    new UnpartitionedRVD(newRowType, crdd.cmapPartitionsWithIndex(f))

  def mapPartitionsWithIndex[T](f: (Int, Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T] = clearingRun(crdd.mapPartitionsWithIndex(f))

  def mapPartitionsWithIndex[T: ClassTag](
    f: (Int, RVDContext, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T] = clearingRun(crdd.cmapPartitionsWithIndex(f))

  def mapPartitions[T](f: (Iterator[RegionValue]) => Iterator[T])(implicit tct: ClassTag[T]): RDD[T] = clearingRun(crdd.mapPartitions(f))

  def mapPartitions[T: ClassTag](
    f: (RVDContext, Iterator[RegionValue]) => Iterator[T]
  ): RDD[T] = clearingRun(crdd.cmapPartitions(f))

  def constrainToOrderedPartitioner(
    ordType: OrderedRVDType,
    newPartitioner: OrderedRVDPartitioner
  ): OrderedRVD

  def treeAggregate[U: ClassTag](zeroValue: U)(
    seqOp: (U, RegionValue) => U,
    combOp: (U, U) => U,
    depth: Int = treeAggDepth(HailContext.get, crdd.getNumPartitions)
  ): U = {
    val clearingSeqOp = { (ctx: RVDContext, u: U, rv: RegionValue) =>
      val u2 = seqOp(u, rv)
      ctx.region.clear()
      u2
    }
    crdd.treeAggregate(zeroValue, clearingSeqOp, combOp, depth)
  }

  def aggregate[U: ClassTag](
    zeroValue: U
  )(seqOp: (U, RegionValue) => U,
    combOp: (U, U) => U
  ): U = {
    val clearingSeqOp = { (ctx: RVDContext, u: U, rv: RegionValue) =>
      val u2 = seqOp(u, rv)
      ctx.region.clear()
      u2
    }
    crdd.aggregate(zeroValue, clearingSeqOp, combOp)
  }

  def count(): Long =
    crdd.cmapPartitions { (ctx, it) =>
      var count = 0L
      it.foreach { rv =>
        count += 1
        ctx.region.clear()
      }
      Iterator.single(count)
    }.run.fold(0L)(_ + _)

  def countPerPartition(): Array[Long] =
    crdd.cmapPartitions { (ctx, it) =>
      var count = 0L
      it.foreach { rv =>
        count += 1
        ctx.region.clear()
      }
      Iterator.single(count)
    }.collect()

  def collectPerPartition[T : ClassTag](f: (RVDContext, Iterator[RegionValue]) => T): Array[T] =
    crdd.cmapPartitions { (ctx, it) =>
      Iterator.single(f(ctx, it))
    }.collect()

  protected def persistRVRDD(level: StorageLevel): PersistedRVRDD = {
    val localRowType = rowType

    val makeEnc = RVD.memoryCodec.buildEncoder(localRowType)

    val makeDec = RVD.memoryCodec.buildDecoder(localRowType, localRowType)

    // copy, persist region values
    val persistedRDD = crdd.cmapPartitions { (ctx, it) =>
      it.map(RVD.regionValueToBytes(makeEnc, ctx))
    } .run
      .persist(level)

    PersistedRVRDD(persistedRDD,
      ContextRDD.weaken[RVDContext](persistedRDD)
        .cmapPartitions { (ctx, it) =>
          val region = ctx.region
          val rv = RegionValue(region)
          it.map(RVD.bytesToRegionValue(makeDec, region, rv))
        })
  }

  def storageLevel: StorageLevel = StorageLevel.NONE

  def persist(level: StorageLevel): RVD

  def cache(): RVD = persist(StorageLevel.MEMORY_ONLY)

  def unpersist(): RVD = this

  def coalesce(maxPartitions: Int, shuffle: Boolean): RVD

  def sample(withReplacement: Boolean, p: Double, seed: Long): RVD

  def zipWithIndex(name: String): RVD

  private[rvd] def zipWithIndexCRDD(
    name: String
  ): (TStruct, ContextRDD[RVDContext, RegionValue]) = {
    val (newRowType, ins) = rowType.unsafeStructInsert(TInt64(), List(name))

    val a = sparkContext.broadcast(countPerPartition().scanLeft(0L)(_ + _))

    val newCRDD = crdd.cmapPartitionsWithIndex({ (i, ctx, it) =>
      val rv2 = RegionValue()
      val rvb = ctx.rvb
      var index = a.value(i)
      it.map { rv =>
        rvb.start(newRowType)
        ins(rv.region, rv.offset, rvb, () => rvb.addLong(index))
        index += 1
        rv2.set(rvb.region, rvb.end())
        rv2
      }
    }, preservesPartitioning=true)
    (newRowType, newCRDD)
  }

  protected def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]): RVDSpec

  final def write(path: String, codecSpec: CodecSpec): Array[Long] = {
    val (partFiles, partitionCounts) = crdd.writeRows(path, rowType, codecSpec)
    rvdSpec(codecSpec, partFiles).write(sparkContext.hadoopConfiguration, path)
    partitionCounts
  }

  def toRows: RDD[Row] = {
    val localRowType = rowType
    map(rv => SafeRow(localRowType, rv.region, rv.offset))
  }

  def toUnpartitionedRVD: UnpartitionedRVD
}
