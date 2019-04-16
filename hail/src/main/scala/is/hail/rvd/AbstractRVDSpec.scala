package is.hail.rvd

import is.hail.HailContext
import is.hail.annotations._
import is.hail.compatibility.UnpartitionedRVDSpec
import is.hail.cxx
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.virtual._
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual.{TStruct, TStructSerializer}
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop
import org.apache.spark.sql.Row
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{DefaultFormats, Formats, JValue, ShortTypeHints}

object AbstractRVDSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[AbstractRVDSpec], classOf[OrderedRVDSpec],
      classOf[CodecSpec], classOf[PackCodecSpec], classOf[BlockBufferSpec],
      classOf[LZ4BlockBufferSpec], classOf[StreamBlockBufferSpec],
      classOf[BufferSpec], classOf[LEB128BufferSpec], classOf[BlockingBufferSpec],
      classOf[UnpartitionedRVDSpec]))
    override val typeHintFieldName = "name"
  } +
    new TStructSerializer +
    new RVDTypeSerializer

  def read(conf: org.apache.hadoop.conf.Configuration, path: String): AbstractRVDSpec = {
    val metadataFile = path + "/metadata.json.gz"
    conf.readFile(metadataFile) { in => JsonMethods.parse(in) }
      .transformField { case ("orvdType", value) => ("rvdType", value) } // ugh
      .extract[AbstractRVDSpec]
  }

  def read(hc: HailContext, path: String): AbstractRVDSpec = read(hc.hadoopConf, path)

  def readLocal(hc: HailContext, path: String, rowType: PStruct, codecSpec: CodecSpec, partFiles: Array[String], requestedType: PStruct): IndexedSeq[Row] = {
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

  def writeLocal(
    hc: HailContext,
    path: String,
    rowType: PStruct,
    codecSpec: CodecSpec,
    rows: IndexedSeq[Annotation]
  ): Array[Long] = {
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
              rvb.addAnnotation(rowType.virtualType, a)
              RegionValue(region, rvb.end())
            }, os)
        }
      }

    val spec = OrderedRVDSpec(
      rowType,
      FastIndexedSeq(),
      codecSpec,
      Array("part-0"),
      RVDPartitioner.unkeyed(1))
    spec.write(hConf, path)

    Array(part0Count)
  }
}

object OrderedRVDSpec {
  def apply(rowType: PStruct,
    key: IndexedSeq[String],
    codecSpec: CodecSpec,
    partFiles: Array[String],
    partitioner: RVDPartitioner): AbstractRVDSpec = {

    OrderedRVDSpec(
      RVDType(rowType, key),
      codecSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(
        partitioner.rangeBounds.toFastSeq,
        partitioner.rangeBoundsType))
  }
}

case class OrderedRVDSpec(
  rvdType: RVDType,
  codecSpec: CodecSpec,
  partFiles: Array[String],
  jRangeBounds: JValue
) extends AbstractRVDSpec {
  def key: IndexedSeq[String] = rvdType.key

  override def encodedType: PStruct = rvdType.rowType

  def partitioner: RVDPartitioner = {
    val rangeBoundsType = TArray(TInterval(rvdType.kType.virtualType))
    new RVDPartitioner(rvdType.kType.virtualType,
      JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }
}

abstract class AbstractRVDSpec {
  def partitioner: RVDPartitioner

  // FIXME introduce EType
  def encodedType: PStruct

  def key: IndexedSeq[String]

  def partFiles: Array[String]

  def codecSpec: CodecSpec

  def read(hc: HailContext, path: String, requestedType: PStruct): RVD = {
    val requestedKey = key.takeWhile(requestedType.hasField)
    val rvdType = RVDType(requestedType, requestedKey)

    RVD(rvdType, partitioner.coarsen(requestedKey.length), hc.readRows(path, encodedType, codecSpec, partFiles, requestedType))
  }

  def readLocal(hc: HailContext, path: String, requestedType: PStruct): IndexedSeq[Row] =
    AbstractRVDSpec.readLocal(hc, path, encodedType, codecSpec, partFiles, requestedType)

  def write(hadoopConf: hadoop.conf.Configuration, path: String) {
    hadoopConf.writeTextFile(path + "/metadata.json.gz") { out =>
      implicit val formats = AbstractRVDSpec.formats
      Serialization.write(this, out)
    }
  }
}
