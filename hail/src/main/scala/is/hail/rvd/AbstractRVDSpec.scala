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
      classOf[AbstractRVDSpec], classOf[OrderedRVDSpec], classOf[IndexedRVDSpec],
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
      val f = partPath(path, p)
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

  def partPath(path: String, partFile: String): String = path + "/parts/" + partFile

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
            }, os, null)
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

  def readZipped(
    hc: HailContext,
    specLeft: AbstractRVDSpec,
    specRight: AbstractRVDSpec,
    pathLeft: String,
    pathRight: String,
    requestedType: PStruct,
    requestedTypeLeft: PStruct,
    requestedTypeRight: PStruct,
    newPartitioner: Option[RVDPartitioner],
    filterIntervals: Boolean
  ): RVD = {
    require(specRight.key.isEmpty)
    require(requestedType == requestedTypeLeft ++ requestedTypeRight)
    val requestedKey = specLeft.key.takeWhile(requestedTypeLeft.hasField)
    val partitioner = specLeft.partitioner
    val tmpPartitioner = partitioner.intersect(newPartitioner.getOrElse(partitioner))

    val rvdType = RVDType(requestedType, requestedKey)
    val parts = if (specLeft.key.isEmpty)
      specLeft.partFiles
    else
      tmpPartitioner.rangeBounds.map { b => specLeft.partFiles(partitioner.lowerBoundInterval(b)) }.toArray

    val (isl, isr) = (specLeft, specRight) match {
      case (l: IndexedRVDSpec, r: IndexedRVDSpec) => (Some(l.indexSpec), Some(r.indexSpec))
      case _ => (None, None)
    }

    val crdd = hc.readRowsSplit(
      pathLeft, pathRight, isl, isr, specLeft.encodedType, specRight.encodedType,
      specLeft.codecSpec, parts, tmpPartitioner.rangeBounds,
      requestedType, requestedTypeLeft, requestedTypeRight)
    val tmprvd = RVD(rvdType, tmpPartitioner.coarsen(requestedKey.length), crdd)
    newPartitioner match {
      case Some(part) if !filterIntervals => tmprvd.repartition(part.coarsen(requestedKey.length))
      case _ => tmprvd
    }
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

  def readIndexed(
    hc: HailContext,
    path: String,
    rowType: PStruct,
    partitioner: Option[RVDPartitioner],
    filterIntervals: Boolean
  ): RVD = partitioner match {
    case Some(_) => fatal("attempted to read unindexed data as indexed")
    case None => read(hc, path, rowType)
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

case class IndexSpec(
  relPath: String,
  keyType: TStruct,
  annotationType: TStruct,
  offsetField: Option[String] = None
) {
  def types: (Type, Type) = (keyType, annotationType)
}

object IndexSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(classOf[IndexSpec]))
    override val typeHintFieldName = "name"
  } + new TStructSerializer

  def emptyAnnotation(relPath: String, keyType: TStruct): IndexSpec =
    IndexSpec(relPath, keyType, (+TStruct()).asInstanceOf[TStruct])

  def defaultAnnotation(relPath: String, keyType: TStruct, withOffsetField: Boolean = false): IndexSpec = {
    val name = "entries_offset"
    IndexSpec(relPath, keyType, (+TStruct(name -> TInt64())).asInstanceOf[TStruct],
      if (withOffsetField) Some(name) else None)
  }
}

object IndexedRVDSpec {
  def apply(rowType: PStruct,
    key: IndexedSeq[String],
    codecSpec: CodecSpec,
    indexSpec: IndexSpec,
    partFiles: Array[String],
    partitioner: RVDPartitioner
  ): AbstractRVDSpec = IndexedRVDSpec(RVDType(rowType, key), codecSpec, indexSpec, partFiles, partitioner)

  def apply(typ: RVDType,
    codecSpec: CodecSpec,
    indexSpec: IndexSpec,
    partFiles: Array[String],
    partitioner: RVDPartitioner
  ): AbstractRVDSpec = {
    IndexedRVDSpec(
      typ,
      codecSpec,
      indexSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(
        partitioner.rangeBounds.toFastSeq,
        partitioner.rangeBoundsType))
  }
}

case class IndexedRVDSpec(
  rvdType: RVDType,
  codecSpec: CodecSpec,
  indexSpec: IndexSpec,
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

  override def readIndexed(
    hc: HailContext,
    path: String,
    requestedType: PStruct,
    newPartitioner: Option[RVDPartitioner],
    filterIntervals: Boolean
  ): RVD = {
    val requestedKey = key.takeWhile(requestedType.hasField)
    val tmpPartitioner = partitioner.intersect(newPartitioner.getOrElse(partitioner))

    val rvdType = RVDType(requestedType, requestedKey)
    val parts = if (key.isEmpty)
      partFiles
    else
      tmpPartitioner.rangeBounds.map { b => partFiles(partitioner.lowerBoundInterval(b)) }.toArray

    val crdd = hc.readIndexedRows(path, indexSpec, encodedType, codecSpec, parts, tmpPartitioner.rangeBounds, requestedType)
    val tmprvd = RVD(rvdType, tmpPartitioner.coarsen(requestedKey.length), crdd)
    newPartitioner match {
      case Some(partitioner) if !filterIntervals => tmprvd.repartition(partitioner.coarsen(requestedKey.length))
      case _ => tmprvd
    }
  }
}
