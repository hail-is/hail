package is.hail.rvd

import is.hail.HailContext
import is.hail.annotations._
import is.hail.compatibility.{Compatibility, RVDSpec_1_1}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.io._
import is.hail.utils._
import is.hail.variant.FileFormat
import org.apache.hadoop
import org.apache.spark.sql.Row
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

object RVDSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[RVDSpec], classOf[RVDSpec_1_1],
      classOf[CodecSpec], classOf[PackCodecSpec], classOf[BlockBufferSpec],
      classOf[LZ4BlockBufferSpec], classOf[StreamBlockBufferSpec],
      classOf[BufferSpec], classOf[LEB128BufferSpec], classOf[BlockingBufferSpec]))
    override val typeHintFieldName = "name"
  } +
    new TStructSerializer +
    new RVDTypeSerializer

  def read(hc: HailContext, path: String): RVDSpec = {
    val metadataFile = path + "/metadata.json.gz"
    val jv = hc.hadoopConf.readFile(metadataFile) { in => JsonMethods.parse(in) }

    Compatibility.extractRVD(jv)
  }

  def readLocal(hc: HailContext, path: String, rowType: TStruct, codecSpec: CodecSpec, partFiles: Array[String], requestedType: TStruct): IndexedSeq[Row] = {
    assert(partFiles.length == 1)

    val hConf = hc.hadoopConf
    partFiles.flatMap { p =>
      val f = path + "/parts/" + p
      hConf.readFile(f) { in =>
        using(RVDContext.default) { ctx =>
          HailContext.readRowsPartition(codecSpec.buildDecoder(rowType.physicalType, requestedType.physicalType))(ctx, in)
            .map { rv =>
              val r = SafeRow(requestedType.physicalType, rv.region, rv.offset)
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
    rowType: TStruct,
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
          RichContextRDDRegionValue.writeRowsPartition(codecSpec.buildEncoder(rowType.physicalType))(ctx,
            rows.iterator.map { a =>
              rvb.start(rowType.physicalType)
              rvb.addAnnotation(rowType, a)
              RegionValue(region, rvb.end())
            }, os)
        }
      }

    val spec = RVDSpec(
      rowType.physicalType,
      FastIndexedSeq(),
      codecSpec,
      Array("part-0"),
      RVDPartitioner.unkeyed(1))
    spec.write(hConf, path)

    Array(part0Count)
  }

  def apply(rowType: PStruct,
    key: IndexedSeq[String],
    codecSpec: CodecSpec,
    partFiles: Array[String],
    partitioner: RVDPartitioner): RVDSpec = {

    RVDSpec_1_1(
      FileFormat.version.rep,
      rowType.virtualType,
      key,
      codecSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(
        partitioner.rangeBounds.toFastSeq,
        partitioner.rangeBoundsType))
  }

}

abstract class RVDSpec {
  def file_version: Int
  def partitioner: RVDPartitioner

  // FIXME introduce EType
  def encodedType: PStruct

  def key: IndexedSeq[String]

  def partFiles: Array[String]

  def codecSpec: CodecSpec

  def read(hc: HailContext, path: String, requestedType: TStruct): RVD = {
    val rvdType = RVDType(requestedType.physicalType, key)

    RVD(rvdType, partitioner, hc.readRows(path, encodedType.virtualType, codecSpec, partFiles, requestedType))
  }

  def readLocal(hc: HailContext, path: String, requestedType: TStruct): IndexedSeq[Row] =
    RVDSpec.readLocal(hc, path, encodedType.virtualType, codecSpec, partFiles, requestedType)

  def write(hadoopConf: hadoop.conf.Configuration, path: String) {
    hadoopConf.writeTextFile(path + "/metadata.json.gz") { out =>
      implicit val formats = RVDSpec.formats
      Serialization.write(this, out)
    }
  }
}
