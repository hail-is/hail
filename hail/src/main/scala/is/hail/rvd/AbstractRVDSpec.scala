package is.hail.rvd

import is.hail.HailContext
import is.hail.annotations._
import is.hail.compatibility.UnpartitionedRVDSpec
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual.{TStructSerializer, _}
import is.hail.io._
import is.hail.utils._
import org.apache.hadoop
import is.hail.io.fs.FS
import org.apache.spark.TaskContext
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
      classOf[StreamBufferSpec], classOf[UnpartitionedRVDSpec]))
    override val typeHintFieldName = "name"
  } +
    new TStructSerializer +
    new RVDTypeSerializer

  def read(fs: is.hail.io.fs.FS, path: String): AbstractRVDSpec = {
    val metadataFile = path + "/metadata.json.gz"
    fs.readFile(metadataFile) { in => JsonMethods.parse(in) }
      .transformField { case ("orvdType", value) => ("rvdType", value) } // ugh
      .extract[AbstractRVDSpec]
  }

  def read(hc: HailContext, path: String): AbstractRVDSpec = read(hc.sFS, path)

  def readLocal(hc: HailContext, path: String, rowType: PStruct, codecSpec: CodecSpec, partFiles: Array[String], requestedType: PStruct): IndexedSeq[Row] = {
    assert(partFiles.length == 1)

    val fs = hc.sFS
    partFiles.flatMap { p =>
      val f = path + "/parts/" + p
      fs.readFile(f) { in =>
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

  def writeSingle(
    fs: is.hail.io.fs.FS,
    path: String,
    rowType: PStruct,
    codecSpec: CodecSpec,
    rows: IndexedSeq[Annotation]
  ): Array[Long] = {
    val partsPath = path + "/parts"
    fs.mkDir(partsPath)

    val filePath = if (TaskContext.get == null)
      "part-0"
    else
      partFile(0, 0, TaskContext.get)

    val part0Count =
      fs.writeFile(partsPath + "/" + filePath) { os =>
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
      Array(filePath),
      RVDPartitioner.unkeyed(1))
    spec.write(fs, path)

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

  def write(fs: FS, path: String) {
    fs.writeTextFile(path + "/metadata.json.gz") { out =>
      implicit val formats = AbstractRVDSpec.formats
      Serialization.write(this, out)
    }
  }
}
