package is.hail.rvd

import is.hail.annotations._
import is.hail.asm4s.AsmFunction3RegionLongLongLong
import is.hail.expr.{JSONAnnotationImpex, ir}
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.{ExecuteContext, IR, PartitionZippedNativeReader}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.index.{InternalNodeBuilder, LeafNodeBuilder}
import is.hail.types.encoded.ETypeSerializer
import is.hail.types.physical.{PCanonicalStruct, PInt64Optional, PStruct, PType, PTypeSerializer}
import is.hail.types.virtual.{TStructSerializer, _}
import is.hail.utils._
import is.hail.{HailContext, compatibility}
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{DefaultFormats, Formats, JValue, ShortTypeHints}

object AbstractRVDSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[AbstractRVDSpec],
      classOf[OrderedRVDSpec2],
      classOf[IndexedRVDSpec2],
      classOf[IndexSpec2],
      classOf[compatibility.OrderedRVDSpec],
      classOf[compatibility.IndexedRVDSpec],
      classOf[compatibility.IndexSpec],
      classOf[compatibility.UnpartitionedRVDSpec],
      classOf[AbstractTypedCodecSpec],
      classOf[TypedCodecSpec]),
    typeHintFieldName = "name") + BufferSpec.shortTypeHints
  }  +
    new TStructSerializer +
    new TypeSerializer +
    new PTypeSerializer +
    new RVDTypeSerializer +
    new ETypeSerializer

  def read(fs: FS, path: String): AbstractRVDSpec = {
    try {
      val metadataFile = path + "/metadata.json.gz"
      using(fs.open(metadataFile)) { in => JsonMethods.parse(in) }
        .transformField { case ("orvdType", value) => ("rvdType", value) } // ugh
        .extract[AbstractRVDSpec]
    } catch {
      case e: Exception => fatal(s"failed to read RVD spec $path", e)
    }
  }

  def readLocal(
    ctx: ExecuteContext,
    path: String,
    enc: AbstractTypedCodecSpec,
    partFiles: Array[String],
    requestedType: TStruct): (PStruct, Long) = {
    assert(partFiles.length == 1)
    val fs = ctx.fs
    val r = ctx.r

    val (rType: PStruct, dec) = enc.buildDecoder(ctx, requestedType)

    val f = partPath(path, partFiles(0))
    using(fs.open(f)) { in =>
      val Array(rv) = HailContext.readRowsPartition(dec)(r, in).toArray
      (rType, rv)
    }
  }

  def partPath(path: String, partFile: String): String = path + "/parts/" + partFile

  def writeSingle(
    execCtx: ExecuteContext,
    path: String,
    rowType: PStruct,
    bufferSpec: BufferSpec,
    rows: IndexedSeq[Annotation]
  ): Array[FileWriteMetadata] = {
    val fs = execCtx.fs
    val partsPath = path + "/parts"
    fs.mkDir(partsPath)

    val filePath = if (TaskContext.get == null)
      "part-0"
    else
      partFile(0, 0, TaskContext.get)
    val codecSpec = TypedCodecSpec(rowType, bufferSpec)

    val (part0Count, bytesWritten) =
      using(fs.create(partsPath + "/" + filePath)) { os =>
        using(RVDContext.default(execCtx.r.pool)) { ctx =>
          val rvb = ctx.rvb
          RichContextRDDRegionValue.writeRowsPartition(codecSpec.buildEncoder(execCtx, rowType))(ctx,
            rows.iterator.map { a =>
              rowType.unstagedStoreJavaObject(a, ctx.r)
            }, os, null)
        }
      }

    val spec = MakeRVDSpec(codecSpec, Array(filePath), RVDPartitioner.unkeyed(1))
    spec.write(fs, path)

    Array(FileWriteMetadata(path, part0Count, bytesWritten))
  }

  def readZippedLowered(
    ctx: ExecuteContext,
    specLeft: AbstractRVDSpec,
    specRight: AbstractRVDSpec,
    pathLeft: String,
    pathRight: String,
    newPartitioner: Option[RVDPartitioner],
    filterIntervals: Boolean,
    requestedType: Type,
    leftRType: TStruct, rightRType: TStruct,
    requestedKey: IndexedSeq[String]
  ): IR => TableStage = {
    require(specRight.key.isEmpty)
    val partitioner = specLeft.partitioner

    val extendedNewPartitioner = newPartitioner.map(_.extendKey(partitioner.kType))
    val tmpPartitioner = extendedNewPartitioner match {
      case Some(np) => np.intersect(partitioner)
      case None => partitioner
    }

    val (indexSpecLeft, indexSpecRight) = (specLeft, specRight) match {
      case (l: Indexed, r: Indexed) if specLeft.key.nonEmpty => (Some(l.indexSpec), Some(r.indexSpec))
      case _ => (None, None)
    }

    val reader = PartitionZippedNativeReader(specLeft.typedCodecSpec, specRight.typedCodecSpec, indexSpecLeft, indexSpecRight, requestedKey)

    val absPathLeft = removeFileProtocol(pathLeft)
    val absPathRight = removeFileProtocol(pathRight)
    val partsAndIntervals: IndexedSeq[(String, Interval)] = if (specLeft.key.isEmpty) {
      specLeft.partFiles.map { p => (p, null) }
    } else {
      val partFiles = specLeft.partFiles
      tmpPartitioner.rangeBounds.map { b => (partFiles(partitioner.lowerBoundInterval(b)), b) }
    }

    val kSize = specLeft.key.size

    val contextsValues: IndexedSeq[Row] = partsAndIntervals.map { case (partPath, interval) =>
      Row(
        s"${ absPathLeft }/parts/${ partPath }",
        s"${ absPathRight }/parts/${ partPath }",
        indexSpecLeft.map(indexSpec => s"${ absPathLeft }/${ indexSpec.relPath }/${ partPath }.idx").orNull,
        RVDPartitioner.intervalToIRRepresentation(interval, kSize))
    }

    val contexts = ir.ToStream(ir.Literal(TArray(reader.contextType), contextsValues))

    val body = (ctx: IR) => ir.ReadPartition(ctx, requestedType, reader)

    { (globals: IR) =>
      val ts = TableStage(
        globals,
        tmpPartitioner,
        TableStageDependency.none,
        contexts,
        body)
      extendedNewPartitioner match {
        case Some(np) if filterIntervals =>
          ts.repartitionNoShuffle(np)
        case _ => ts
      }
    }
  }

  def readZipped(
    ctx: ExecuteContext,
    specLeft: AbstractRVDSpec,
    specRight: AbstractRVDSpec,
    pathLeft: String,
    pathRight: String,
    newPartitioner: Option[RVDPartitioner],
    filterIntervals: Boolean,
    requestedType: Type,
    leftRType: TStruct, rightRType: TStruct,
    requestedKey: IndexedSeq[String],
    fieldInserter: (ExecuteContext, PStruct, PStruct) => (PStruct, (FS, Int, Region) => AsmFunction3RegionLongLongLong)
  ): RVD = {
    require(specRight.key.isEmpty)
    val partitioner = specLeft.partitioner

    val extendedNewPartitioner = newPartitioner.map(_.extendKey(partitioner.kType))
    val tmpPartitioner = extendedNewPartitioner match {
      case Some(np) => np.intersect(partitioner)
      case None => partitioner
    }

    val parts = if (specLeft.key.isEmpty)
      specLeft.partFiles
    else
      tmpPartitioner.rangeBounds.map { b => specLeft.partFiles(partitioner.lowerBoundInterval(b)) }

    val (isl, isr) = (specLeft, specRight) match {
      case (l: Indexed, r: Indexed) => (Some(l.indexSpec), Some(r.indexSpec))
      case _ => (None, None)
    }

    val (leftPType: PStruct, makeLeftDec) = specLeft.typedCodecSpec.buildDecoder(ctx, leftRType)
    val (rightPType: PStruct, makeRightDec) = specRight.typedCodecSpec.buildDecoder(ctx, rightRType)

    val (t: PStruct, makeInserter) = fieldInserter(ctx, leftPType, rightPType)
    assert(t.virtualType == requestedType)
    val crdd = HailContext.readRowsSplit(ctx,
      pathLeft, pathRight, isl, isr,
      parts, tmpPartitioner.rangeBounds,
      makeLeftDec, makeRightDec, makeInserter)
    val tmprvd = RVD(RVDType(t, requestedKey), tmpPartitioner.coarsen(requestedKey.length), crdd)
    extendedNewPartitioner match {
      case Some(part) if !filterIntervals => tmprvd.repartition(ctx, part.coarsen(requestedKey.length))
      case _ => tmprvd
    }
  }
}

trait Indexed extends AbstractRVDSpec {
  override val indexed: Boolean = true

  def indexSpec: AbstractIndexSpec
}

abstract class AbstractRVDSpec {
  def partitioner: RVDPartitioner

  def key: IndexedSeq[String]

  def partFiles: Array[String]

  def absolutePartPaths(path: String): Array[String] = partFiles.map(path + "/parts/" + _)

  def typedCodecSpec: AbstractTypedCodecSpec

  def indexed: Boolean = false

  def attrs: Map[String, String]

  def read(
    ctx: ExecuteContext,
    path: String,
    requestedType: TStruct,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): RVD = newPartitioner match {
    case Some(_) => fatal("attempted to read unindexed data as indexed")
    case None =>
      val requestedKey = key.takeWhile(requestedType.hasField)
      val (pType: PStruct, crdd) = HailContext.readRows(ctx, path, typedCodecSpec, partFiles, requestedType)
      val rvdType = RVDType(pType, requestedKey)

      RVD(rvdType, partitioner.coarsen(requestedKey.length), crdd)
  }

  def readTableStage(
    ctx: ExecuteContext,
    path: String,
    requestedType: TStruct,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): IR => TableStage = newPartitioner match {
    case Some(_) => fatal("attempted to read unindexed data as indexed")
    case None =>

      val rSpec = typedCodecSpec

      val ctxType = TStruct("path" -> TString)
      val contexts = ir.ToStream(ir.Literal(TArray(ctxType), absolutePartPaths(path).map(x => Row(x)).toFastIndexedSeq))

      val body = (ctx: IR) => ir.ReadPartition(ir.GetField(ctx, "path"), requestedType, ir.PartitionNativeReader(rSpec))

      (globals: IR) =>
        TableStage(
          globals,
          partitioner,
          TableStageDependency.none,
          contexts,
          body)
  }

  def readLocalSingleRow(ctx: ExecuteContext, path: String, requestedType: TStruct): (PStruct, Long) =
    AbstractRVDSpec.readLocal(ctx, path, typedCodecSpec, partFiles, requestedType)

  def write(fs: FS, path: String) {
    using(fs.create(path + "/metadata.json.gz")) { out =>
      import AbstractRVDSpec.formats
      Serialization.write(this, out)
    }
  }
}

trait AbstractIndexSpec {
  def relPath: String

  def leafCodec: AbstractTypedCodecSpec

  def internalNodeCodec: AbstractTypedCodecSpec

  def keyType: Type

  def annotationType: Type

  def offsetField: Option[String] = None

  def types: (Type, Type) = (keyType, annotationType)
}

case class IndexSpec2(_relPath: String,
  _leafCodec: AbstractTypedCodecSpec,
  _internalNodeCodec: AbstractTypedCodecSpec,
  _keyType: Type,
  _annotationType: Type,
  _offsetField: Option[String] = None
) extends AbstractIndexSpec {
  def relPath: String = _relPath

  def leafCodec: AbstractTypedCodecSpec = _leafCodec

  def internalNodeCodec: AbstractTypedCodecSpec = _internalNodeCodec

  def keyType: Type = _keyType

  def annotationType: Type = _annotationType

  override def offsetField: Option[String] = _offsetField
}


object IndexSpec {

  def fromKeyAndValuePTypes(relPath: String, keyPType: PType, annotationPType: PType, offsetFieldName: Option[String]): AbstractIndexSpec = {
    val leafType = LeafNodeBuilder.typ(keyPType, annotationPType)
    val leafNodeSpec = TypedCodecSpec(leafType, BufferSpec.default)
    val internalType = InternalNodeBuilder.typ(keyPType, annotationPType)
    val internalNodeSpec = TypedCodecSpec(internalType, BufferSpec.default)
    IndexSpec2(relPath, leafNodeSpec, internalNodeSpec, keyPType.virtualType, annotationPType.virtualType, offsetFieldName)
  }

  def emptyAnnotation(relPath: String, keyType: PStruct): AbstractIndexSpec = {
    fromKeyAndValuePTypes(relPath, keyType, PCanonicalStruct(required = true), None)
  }

  def defaultAnnotation(relPath: String, keyType: PStruct, withOffsetField: Boolean = false): AbstractIndexSpec = {
    val name = "entries_offset"
    fromKeyAndValuePTypes(relPath, keyType, PCanonicalStruct(required = true, name -> PInt64Optional),
      if (withOffsetField) Some(name) else None)
  }
}

object MakeRVDSpec {
  def apply(
    codecSpec: AbstractTypedCodecSpec,
    partFiles: Array[String],
    partitioner: RVDPartitioner,
    indexSpec: AbstractIndexSpec = null,
    attrs: Map[String, String] = Map.empty
  ): AbstractRVDSpec =
    RVDSpecMaker(codecSpec, partitioner, indexSpec, attrs)(partFiles)
}

object RVDSpecMaker {
  def apply(codecSpec: AbstractTypedCodecSpec,
    partitioner: RVDPartitioner,
    indexSpec: AbstractIndexSpec = null,
    attrs: Map[String, String] =  Map.empty): RVDSpecMaker = RVDSpecMaker(
    codecSpec,
    partitioner.kType.fieldNames,
    JSONAnnotationImpex.exportAnnotation(
      partitioner.rangeBounds.toFastSeq,
      partitioner.rangeBoundsType),
    indexSpec,
    attrs)
}

case class RVDSpecMaker(
  codecSpec: AbstractTypedCodecSpec,
  key: IndexedSeq[String],
  bounds: JValue,
  indexSpec: AbstractIndexSpec,
  attrs: Map[String, String]) {
  def apply(partFiles: Array[String]): AbstractRVDSpec =
    Option(indexSpec) match {
      case Some(ais) => IndexedRVDSpec2(
        key,
        codecSpec,
        ais,
        partFiles,
        bounds,
        attrs)
      case None => OrderedRVDSpec2(
        key,
        codecSpec,
        partFiles,
        bounds,
        attrs
      )
    }
}

object IndexedRVDSpec2 {
  def apply(key: IndexedSeq[String],
    codecSpec: AbstractTypedCodecSpec,
    indexSpec: AbstractIndexSpec,
    partFiles: Array[String],
    partitioner: RVDPartitioner,
    attrs: Map[String, String]
  ): AbstractRVDSpec = {
    IndexedRVDSpec2(
      key,
      codecSpec,
      indexSpec,
      partFiles,
      JSONAnnotationImpex.exportAnnotation(
        partitioner.rangeBounds.toFastSeq,
        partitioner.rangeBoundsType),
      attrs)
  }
}

case class IndexedRVDSpec2(_key: IndexedSeq[String],
  _codecSpec: AbstractTypedCodecSpec,
  _indexSpec: AbstractIndexSpec,
  _partFiles: Array[String],
  _jRangeBounds: JValue,
  _attrs: Map[String, String]) extends AbstractRVDSpec with Indexed {

  // some lagacy OrderedRVDSpec2 were written out without the toplevel encoder required
  private val codecSpec2 = _codecSpec match {
    case cs: TypedCodecSpec =>
      TypedCodecSpec(cs._eType.setRequired(true), cs._vType, cs._bufferSpec)
  }

  require(codecSpec2.encodedType.required)

  def typedCodecSpec: AbstractTypedCodecSpec = codecSpec2

  def indexSpec: AbstractIndexSpec = _indexSpec

  lazy val partitioner: RVDPartitioner = {
    val keyType = codecSpec2.encodedVirtualType.asInstanceOf[TStruct].select(key)._1
    val rangeBoundsType = TArray(TInterval(keyType))
    new RVDPartitioner(keyType,
      JSONAnnotationImpex.importAnnotation(_jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  def partFiles: Array[String] = _partFiles

  def key: IndexedSeq[String] = _key

  val attrs: Map[String, String] = _attrs

  override def read(
    ctx: ExecuteContext,
    path: String,
    requestedType: TStruct,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): RVD = {
    newPartitioner match {
      case Some(np) =>
        val extendedNP = np.extendKey(partitioner.kType)
        val requestedKey = key.takeWhile(requestedType.hasField)
        val tmpPartitioner = partitioner.intersect(extendedNP)

        assert(key.nonEmpty)
        val parts = tmpPartitioner.rangeBounds.map { b => partFiles(partitioner.lowerBoundInterval(b)) }

        val (decPType: PStruct, crdd) = HailContext.readIndexedRows(ctx, path, _indexSpec, typedCodecSpec, parts, tmpPartitioner.rangeBounds, requestedType)
        val rvdType = RVDType(decPType, requestedKey)
        val tmprvd = RVD(rvdType, tmpPartitioner.coarsen(requestedKey.length), crdd)

        if (filterIntervals)
          tmprvd
        else
          tmprvd.repartition(ctx, extendedNP.coarsen(requestedKey.length))
      case None =>
        // indexed reads are costly; don't use an indexed read when possible
        super.read(ctx, path, requestedType, None, filterIntervals)
    }
  }

  override def readTableStage(
    ctx: ExecuteContext,
    path: String,
    requestedType: TStruct,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): IR => TableStage = newPartitioner match {
    case Some(np) =>

      val extendedNP = np.extendKey(partitioner.kType)
      val tmpPartitioner = partitioner.intersect(extendedNP)

      assert(key.nonEmpty)

      val rSpec = typedCodecSpec
      val reader = ir.PartitionNativeReaderIndexed(rSpec, indexSpec, partitioner.kType.fieldNames)

      val absPath = removeFileProtocol(path)
      val partPaths = tmpPartitioner.rangeBounds.map { b => partFiles(partitioner.lowerBoundInterval(b)) }


      val kSize = partitioner.kType.size
      absolutePartPaths(path)
      assert(tmpPartitioner.rangeBounds.size == partPaths.length)
      val contextsValues: IndexedSeq[Row] = tmpPartitioner.rangeBounds.zip(partPaths).map { case (interval, partPath) =>
        Row(
          s"${ absPath }/parts/${ partPath }",
          s"${ absPath }/${ indexSpec.relPath }/${ partPath }.idx",
          RVDPartitioner.intervalToIRRepresentation(interval, kSize))
      }

      assert(TArray(reader.contextType).typeCheck(contextsValues))

      val contexts = ir.ToStream(ir.Literal(TArray(reader.contextType), contextsValues))

      val body = (ctx: IR) => ir.ReadPartition(ctx, requestedType, reader)

      { (globals: IR) =>
        val ts = TableStage(
          globals,
          tmpPartitioner,
          TableStageDependency.none,
          contexts,
          body)
        if (filterIntervals) ts else ts.repartitionNoShuffle(extendedNP)
      }

    case None =>
      super.readTableStage(ctx, path, requestedType, newPartitioner, filterIntervals)
  }
}

case class OrderedRVDSpec2(_key: IndexedSeq[String],
  _codecSpec: AbstractTypedCodecSpec,
  _partFiles: Array[String],
  _jRangeBounds: JValue,
  _attrs: Map[String, String]) extends AbstractRVDSpec {

  // some legacy OrderedRVDSpec2 were written out without the toplevel encoder required
  private val codecSpec2 = _codecSpec match {
    case cs: TypedCodecSpec =>
      TypedCodecSpec(cs._eType.setRequired(true), cs._vType, cs._bufferSpec)
  }

  require(codecSpec2.encodedType.required)

  lazy val partitioner: RVDPartitioner = {
    val keyType = codecSpec2.encodedVirtualType.asInstanceOf[TStruct].select(key)._1
    val rangeBoundsType = TArray(TInterval(keyType))
    new RVDPartitioner(keyType,
      JSONAnnotationImpex.importAnnotation(_jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  def partFiles: Array[String] = _partFiles

  def key: IndexedSeq[String] = _key

  val attrs: Map[String, String] = _attrs

  def typedCodecSpec: AbstractTypedCodecSpec = codecSpec2
}
