package is.hail.rvd

import is.hail.HailContext
import is.hail.annotations._
import is.hail.compatibility.UnpartitionedRVDSpec
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.physical.{PStruct, PTypeSerializer}
import is.hail.expr.types.virtual.{TStructSerializer, _}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.index.{InternalNodeBuilder, LeafNodeBuilder}
import is.hail.utils._
import org.apache.spark.TaskContext
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{DefaultFormats, Formats, JValue, ShortTypeHints}

object AbstractRVDSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[AbstractRVDSpec], classOf[OrderedRVDSpec], classOf[IndexedRVDSpec],
      classOf[CodecSpec], classOf[PackCodecSpec], classOf[BlockBufferSpec],
      classOf[LZ4BlockBufferSpec], classOf[StreamBlockBufferSpec],
      classOf[BufferSpec], classOf[LEB128BufferSpec], classOf[BlockingBufferSpec],
      classOf[StreamBufferSpec], classOf[UnpartitionedRVDSpec],
      classOf[CodecSpec2], classOf[PackCodecSpec2]))
    override val typeHintFieldName = "name"
  } +
    new TStructSerializer +
    new TypeSerializer +
    new PTypeSerializer +
    new RVDTypeSerializer

  def read(fs: is.hail.io.fs.FS, path: String): AbstractRVDSpec = {
    val metadataFile = path + "/metadata.json.gz"
    fs.readFile(metadataFile) { in => JsonMethods.parse(in) }
      .transformField { case ("orvdType", value) => ("rvdType", value) } // ugh
      .extract[AbstractRVDSpec]
  }

  def read(hc: HailContext, path: String): AbstractRVDSpec = read(hc.sFS, path)

  def readLocal(hc: HailContext,
    path: String,
    enc: CodecSpec2,
    partFiles: Array[String],
    requestedType: TStruct,
    r: Region): (PStruct, Long) = {
    assert(partFiles.length == 1)

    val fs = hc.sFS

    val (rType: PStruct, dec) = enc.buildDecoder(requestedType)

    val f = partPath(path, partFiles(0))
    fs.readFile(f) { in =>
      val Array(rv) = HailContext.readRowsPartition(dec)(r, in).toArray
      (rType, rv.offset)
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
          RichContextRDDRegionValue.writeRowsPartition(PackCodecSpec2(rowType, codecSpec.asInstanceOf[PackCodecSpec].child).buildEncoder(rowType))(ctx,
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
    requestedType: TStruct,
    requestedTypeLeft: TStruct,
    requestedTypeRight: TStruct,
    newPartitioner: Option[RVDPartitioner],
    filterIntervals: Boolean
  ): RVD = {
    require(specRight.key.isEmpty)
    require(requestedType == requestedTypeLeft ++ requestedTypeRight)
    val requestedKey = specLeft.key.takeWhile(requestedTypeLeft.hasField)
    val partitioner = specLeft.partitioner
    val tmpPartitioner = partitioner.intersect(newPartitioner.getOrElse(partitioner))

    val parts = if (specLeft.key.isEmpty)
      specLeft.partFiles
    else
      tmpPartitioner.rangeBounds.map { b => specLeft.partFiles(partitioner.lowerBoundInterval(b)) }.toArray

    val (isl, isr) = (specLeft, specRight) match {
      case (l: IndexedRVDSpec, r: IndexedRVDSpec) => (Some(l.indexSpec), Some(r.indexSpec))
      case _ => (None, None)
    }

    val (t, crdd) = hc.readRowsSplit(
      pathLeft, pathRight, isl, isr,
      specLeft.codecSpec2, specRight.codecSpec2, parts, tmpPartitioner.rangeBounds,
      requestedTypeLeft, requestedTypeRight)
    assert(t.virtualType == requestedType)
    val tmprvd = RVD(RVDType(t, requestedKey), tmpPartitioner.coarsen(requestedKey.length), crdd)
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

  override def encodedType: TStruct = rvdType.rowType.virtualType

  def partitioner: RVDPartitioner = {
    val rangeBoundsType = TArray(TInterval(rvdType.kType.virtualType))
    new RVDPartitioner(rvdType.kType.virtualType,
      JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  override def codecSpec2: CodecSpec2 = codecSpec.makeCodecSpec2(rvdType.rowType)
}

abstract class AbstractRVDSpec {
  def partitioner: RVDPartitioner

  // FIXME introduce EType
  def encodedType: TStruct

  def key: IndexedSeq[String]

  def partFiles: Array[String]

  def codecSpec2: CodecSpec2

  val indexed: Boolean = false

  def read(
    hc: HailContext,
    path: String,
    requestedType: TStruct,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): RVD = newPartitioner match {
    case Some(_) => fatal("attempted to read unindexed data as indexed")
    case None =>
      val requestedKey = key.takeWhile(requestedType.hasField)
      val (pType: PStruct, crdd) = hc.readRows(path, codecSpec2, partFiles, requestedType)
      val rvdType = RVDType(pType, requestedKey)

      RVD(rvdType, partitioner.coarsen(requestedKey.length), crdd)
  }

  def readLocalSingleRow(hc: HailContext, path: String, requestedType: TStruct, r: Region): (PStruct, Long) =
    AbstractRVDSpec.readLocal(hc, path, codecSpec2, partFiles, requestedType, r)

  def write(fs: FS, path: String) {
    fs.writeTextFile(path + "/metadata.json.gz") { out =>
      implicit val formats = AbstractRVDSpec.formats
      Serialization.write(this, out)
    }
  }
}

trait AbstractIndexSpec {
  def relPath: String
  def leafCodec: CodecSpec2
  def internalNodeCodec: CodecSpec2
  def keyType: Type
  def annotationType: Type
  def offsetField: Option[String] = None
  def types: (Type, Type) = (keyType, annotationType)
}

case class IndexSpec(
  relPath: String,
  keyType: TStruct,
  annotationType: TStruct,
  override val offsetField: Option[String] = None
) extends AbstractIndexSpec {
  val baseSpec = LEB128BufferSpec(
      BlockingBufferSpec(32 * 1024,
        LZ4BlockBufferSpec(32 * 1024,
          new StreamBlockBufferSpec)))

  def leafCodec: CodecSpec2 = PackCodecSpec2(LeafNodeBuilder.typ(keyType.physicalType, annotationType.physicalType), baseSpec)

  def internalNodeCodec: CodecSpec2 = PackCodecSpec2(InternalNodeBuilder.typ(keyType.physicalType, annotationType.physicalType), baseSpec)
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

  override def encodedType: TStruct = rvdType.rowType.virtualType

  val partitioner: RVDPartitioner = {
    val rangeBoundsType = TArray(TInterval(rvdType.kType.virtualType))
    new RVDPartitioner(rvdType.kType.virtualType,
      JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  override val indexed = true

  override def read(
    hc: HailContext,
    path: String,
    requestedType: TStruct,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): RVD = {
    newPartitioner match {
      case Some(np) =>
        val requestedKey = key.takeWhile(requestedType.hasField)
        val tmpPartitioner = partitioner.intersect(np)

        assert(key.nonEmpty)
        val parts = tmpPartitioner.rangeBounds.map { b => partFiles(partitioner.lowerBoundInterval(b)) }

        val (decPType: PStruct, crdd) = hc.readIndexedRows(path, indexSpec, codecSpec2, parts, tmpPartitioner.rangeBounds, requestedType)
        val rvdType = RVDType(decPType, requestedKey)
        val tmprvd = RVD(rvdType, tmpPartitioner.coarsen(requestedKey.length), crdd)

        if (filterIntervals)
          tmprvd
        else
          tmprvd.repartition(np.coarsen(requestedKey.length))
      case None =>
        // indexed reads are costly; don't use an indexed read when possible
        super.read(hc, path, requestedType, None, filterIntervals)
    }
  }

  override def codecSpec2: CodecSpec2 = codecSpec.makeCodecSpec2(rvdType.rowType)
}
