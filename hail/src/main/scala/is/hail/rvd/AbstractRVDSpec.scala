package is.hail.rvd

import cats.implicits.{toFlatMapOps, toFunctorOps}
import is.hail.annotations._
import is.hail.expr.ir.lowering.utils.assertA
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.compatibility
import is.hail.expr.ir.lowering.{MonadLower, TableStage, TableStageDependency}
import is.hail.expr.ir.{IR, Literal, PartitionNativeReader, PartitionZippedIndexedNativeReader, PartitionZippedNativeReader, ReadPartition, ToStream}
import is.hail.expr.{JSONAnnotationImpex, ir}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.index.{InternalNodeBuilder, LeafNodeBuilder}
import is.hail.types.TableType
import is.hail.types.encoded.ETypeSerializer
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{DefaultFormats, Formats, JValue, ShortTypeHints}

import scala.language.higherKinds

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
  } +
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

  def partPath(path: String, partFile: String): String = path + "/parts/" + partFile

  def writeSingle(execCtx: ExecuteContext,
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
          RichContextRDDRegionValue.writeRowsPartition(codecSpec.buildEncoder(execCtx, rowType))(
            ctx,
            rows.iterator.map { a =>
              rowType.unstagedStoreJavaObject(execCtx.stateManager, a, ctx.r)
            }, os, null)
        }
      }

    val spec = MakeRVDSpec(codecSpec, Array(filePath), RVDPartitioner.unkeyed(execCtx.stateManager, 1))
    spec.write(fs, path)

    Array(FileWriteMetadata(path, part0Count, bytesWritten))
  }

  def readZippedLowered[M[_]](specLeft: AbstractRVDSpec,
                              specRight: AbstractRVDSpec,
                              pathLeft: String,
                              pathRight: String,
                              newPartitioner: Option[RVDPartitioner],
                              filterIntervals: Boolean,
                              requestedType: TStruct,
                              requestedKey: IndexedSeq[String],
                              uidFieldName: String
                             )
                             (globals: IR)
                             (implicit M: MonadLower[M]): M[TableStage] = {
    require(specRight.key.isEmpty)
    val readPartitioner = M.reader(c => specLeft.partitioner(c.stateManager))

    newPartitioner match {
      case None =>
        val reader = PartitionZippedNativeReader(
          PartitionNativeReader(specLeft.typedCodecSpec, uidFieldName),
          PartitionNativeReader(specRight.typedCodecSpec, uidFieldName)
        )

        val leftParts = specLeft.absolutePartPaths(pathLeft)
        val rightParts = specRight.absolutePartPaths(pathRight)
        assert(leftParts.length == rightParts.length)
        val contextsValue: IndexedSeq[Any] = (leftParts, rightParts, leftParts.indices)
          .zipped
          .map { (path1, path2, partIdx) => Row(Row(partIdx.toLong, path1), Row(partIdx.toLong, path2)) }

        val ctxIR = ToStream(Literal(TArray(reader.contextType), contextsValue))
        readPartitioner.map { p =>
          val partKeyPrefix = p.kType.fieldNames.slice(0, requestedKey.length).toIndexedSeq
          assert(requestedKey == partKeyPrefix, s"$requestedKey != $partKeyPrefix")
          TableStage(
            globals,
            p.coarsen(requestedKey.length),
            TableStageDependency.none,
            ctxIR,
            ReadPartition(_, requestedType, reader))
        }

      case Some(np) =>
        for {
          (indexSpecLeft, indexSpecRight) <- (specLeft, specRight) match {
            case (l: Indexed, r: Indexed) => M.pure((l.indexSpec, r.indexSpec))
            case _ => M.raiseError(new RuntimeException(s"attempted to read unindexed table as indexed"))
          }

          _ <- assertA(requestedKey.isEmpty, "cannot read indexed matrix with empty key")
          partitioner <- readPartitioner
          extendedNewPartitioner = np.extendKey(partitioner.kType)
          tmpPartitioner = extendedNewPartitioner.intersect(partitioner)

          partKeyPrefix = tmpPartitioner.kType.fieldNames.slice(0, requestedKey.length).toIndexedSeq
          _ <- assertA(requestedKey == partKeyPrefix, s"$requestedKey != $partKeyPrefix")

          reader = PartitionZippedIndexedNativeReader(
            specLeft.typedCodecSpec, specRight.typedCodecSpec,
            indexSpecLeft, indexSpecRight,
            specLeft.key, uidFieldName
          )

          absPathLeft = removeFileProtocol(pathLeft)
          absPathRight = removeFileProtocol(pathRight)
          partsAndIntervals =
            if (specLeft.key.isEmpty) specLeft.partFiles.map { p => (p, null) }
            else {
              val partFiles = specLeft.partFiles
              tmpPartitioner.rangeBounds.map { b => (partFiles(partitioner.lowerBoundInterval(b)), b) }
            }

          kSize = specLeft.key.size
          contextsValues =
            partsAndIntervals.zipWithIndex.map { case ((partPath, interval), partIdx) =>
              Row(
                partIdx.toLong,
                s"${absPathLeft}/parts/${partPath}",
                s"${absPathRight}/parts/${partPath}",
                s"${absPathLeft}/${indexSpecLeft.relPath}/${partPath}.idx",
                RVDPartitioner.intervalToIRRepresentation(interval, kSize)
              )
            }

          ts = TableStage(
            globals,
            tmpPartitioner.coarsen(requestedKey.length),
            TableStageDependency.none,
            ir.ToStream(ir.Literal(TArray(reader.contextType), contextsValues.toIndexedSeq)),
            (ctx: IR) => ir.ReadPartition(ctx, requestedType, reader)
          )

          tableStage <-
            if (filterIntervals) ts.repartitionNoShuffle(partitioner, dropEmptyPartitions = true)
            else ts.repartitionNoShuffle(extendedNewPartitioner.coarsen(requestedKey.length))
        } yield tableStage
    }
  }
}

trait Indexed extends AbstractRVDSpec {
  override val indexed: Boolean = true

  def indexSpec: AbstractIndexSpec
}

abstract class AbstractRVDSpec {
  def partitioner(sm: HailStateManager): RVDPartitioner

  def key: IndexedSeq[String]

  def partFiles: Array[String]

  def absolutePartPaths(path: String): Array[String] = partFiles.map(path + "/parts/" + _)

  def typedCodecSpec: AbstractTypedCodecSpec

  def indexed: Boolean = false

  def attrs: Map[String, String]

  def readTableStage[M[_]](path: String,
                           requestedType: TableType,
                           uidFieldName: String,
                           newPartitioner: Option[RVDPartitioner] = None,
                           filterIntervals: Boolean = false
                          )(globals: IR)
                          (implicit M: MonadLower[M]): M[TableStage] =
    newPartitioner match {
      case Some(_) => M.raiseError(new HailException("attempted to read unindexed data as indexed"))
      case None =>
        M.reader { ctx =>
          val part = partitioner(ctx.stateManager)
          if (!part.kType.fieldNames.startsWith(requestedType.key))
            fatal(s"Error while reading table ${path}: legacy table written without key." +
              s"\n  Read and write with version 0.2.70 or earlier")

          val rSpec = typedCodecSpec

          val ctxType = TStruct("partitionIndex" -> TInt64, "partitionPath" -> TString)
          val contexts = ir.ToStream(ir.Literal(
            TArray(ctxType),
            absolutePartPaths(path).zipWithIndex.map {
              case (x, i) => Row(i.toLong, x)
            }.toFastIndexedSeq))

          val body = (ctx: IR) =>
            ir.ReadPartition(ctx, requestedType.rowType, ir.PartitionNativeReader(rSpec, uidFieldName))

          TableStage(
            globals,
            part.coarsen(part.kType.fieldNames.takeWhile(requestedType.rowType.hasField).length),
            TableStageDependency.none,
            contexts,
            body
          )
        }
    }

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

  def offsetFieldIndex: Option[Int] = offsetField.map(f => annotationType.asInstanceOf[TStruct].fieldIdx(f))

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

case class IndexedRVDSpec2(
  _key: IndexedSeq[String],
  _codecSpec: AbstractTypedCodecSpec,
  _indexSpec: AbstractIndexSpec,
  _partFiles: Array[String],
  _jRangeBounds: JValue,
  _attrs: Map[String, String]
) extends AbstractRVDSpec with Indexed {

  // some lagacy OrderedRVDSpec2 were written out without the toplevel encoder required
  private val codecSpec2 = _codecSpec match {
    case cs: TypedCodecSpec =>
      TypedCodecSpec(cs._eType.setRequired(true), cs._vType, cs._bufferSpec)
  }

  require(codecSpec2.encodedType.required)

  def typedCodecSpec: AbstractTypedCodecSpec = codecSpec2

  def indexSpec: AbstractIndexSpec = _indexSpec

  def partitioner(sm: HailStateManager): RVDPartitioner = {
    val keyType = codecSpec2.encodedVirtualType.asInstanceOf[TStruct].select(key)._1
    val rangeBoundsType = TArray(TInterval(keyType))
    new RVDPartitioner(sm, keyType,
      JSONAnnotationImpex.importAnnotation(_jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  def partFiles: Array[String] = _partFiles

  def key: IndexedSeq[String] = _key

  val attrs: Map[String, String] = _attrs

  override def readTableStage[M[_]](path: String,
                                    requestedType: TableType,
                                    uidFieldName: String,
                                    newPartitioner: Option[RVDPartitioner] = None,
                                    filterIntervals: Boolean = false
                                   )
                                   (globals: IR)
                                   (implicit M: MonadLower[M]):
  M[TableStage] =
    newPartitioner match {
      case Some(np) =>
        M.ask.flatMap { ctx =>
          val part = partitioner(ctx.stateManager)
          val extendedNP = np.extendKey(part.kType)
          val tmpPartitioner = part.intersect(extendedNP)

          assert(key.nonEmpty)

          val rSpec = typedCodecSpec
          val reader = ir.PartitionNativeReaderIndexed(rSpec, indexSpec, part.kType.fieldNames, uidFieldName)

          val absPath = removeFileProtocol(path)
          val partPaths = tmpPartitioner.rangeBounds.map { b => partFiles(part.lowerBoundInterval(b)) }


          val kSize = part.kType.size
          absolutePartPaths(path)
          assert(tmpPartitioner.rangeBounds.size == partPaths.length)
          val contextsValues: IndexedSeq[Row] = tmpPartitioner.rangeBounds.map { interval =>
            val partIdx = part.lowerBoundInterval(interval)
            val partPath = partFiles(partIdx)
            Row(
              partIdx.toLong,
              s"${absPath}/parts/${partPath}",
              s"${absPath}/${indexSpec.relPath}/${partPath}.idx",
              RVDPartitioner.intervalToIRRepresentation(interval, kSize))
          }

          assert(TArray(reader.contextType).typeCheck(contextsValues))
          val contexts = ir.ToStream(ir.Literal(TArray(reader.contextType), contextsValues))
          val body = (ctx: IR) => ir.ReadPartition(ctx, requestedType.rowType, reader)
          val ts = TableStage(
            globals,
            tmpPartitioner,
            TableStageDependency.none,
            contexts,
            body)
          if (filterIntervals) ts.repartitionNoShuffle(part, dropEmptyPartitions = true)
          else ts.repartitionNoShuffle(extendedNP)
        }

      case None =>
        super.readTableStage(path, requestedType, uidFieldName, newPartitioner, filterIntervals)(globals)
    }
}

case class OrderedRVDSpec2(
  _key: IndexedSeq[String],
  _codecSpec: AbstractTypedCodecSpec,
  _partFiles: Array[String],
  _jRangeBounds: JValue,
  _attrs: Map[String, String]
) extends AbstractRVDSpec {

  // some legacy OrderedRVDSpec2 were written out without the toplevel encoder required
  private val codecSpec2 = _codecSpec match {
    case cs: TypedCodecSpec =>
      TypedCodecSpec(cs._eType.setRequired(true), cs._vType, cs._bufferSpec)
  }

  require(codecSpec2.encodedType.required)

  def partitioner(sm: HailStateManager): RVDPartitioner = {
    val keyType = codecSpec2.encodedVirtualType.asInstanceOf[TStruct].select(key)._1
    val rangeBoundsType = TArray(TInterval(keyType))
    new RVDPartitioner(sm, keyType,
      JSONAnnotationImpex.importAnnotation(_jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  def partFiles: Array[String] = _partFiles

  def key: IndexedSeq[String] = _key

  def attrs: Map[String, String] = _attrs

  def typedCodecSpec: AbstractTypedCodecSpec = codecSpec2
}
