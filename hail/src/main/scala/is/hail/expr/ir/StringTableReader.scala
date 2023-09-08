package is.hail.expr.ir
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage, TableStageDependency, TableStageToRVD}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.{FS, FileListEntry}
import is.hail.rvd.RVDPartitioner
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SJavaString, SStackStruct, SStackStructValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SStreamValue}
import is.hail.types.physical.stypes.primitives.{SInt64, SInt64Value}
import is.hail.types.virtual._
import is.hail.types.{BaseTypeWithRequiredness, RStruct, TableType, VirtualTypeWithReq}
import is.hail.utils.{FastIndexedSeq, FastSeq, fatal, checkGzipOfGlobbedFiles}
import org.json4s.{Extraction, Formats, JValue}

case class StringTableReaderParameters(
  files: Array[String],
  minPartitions: Option[Int],
  forceBGZ: Boolean,
  forceGZ: Boolean,
  filePerPartition: Boolean)

object StringTableReader {
  def apply(fs: FS, params: StringTableReaderParameters): StringTableReader = {
    val fileListEntries = fs.globAll(params.files)
    checkGzipOfGlobbedFiles(params.files, fileListEntries, params.forceGZ, params.forceBGZ)
    new StringTableReader(params, fileListEntries)
  }
  def fromJValue(fs: FS, jv: JValue): StringTableReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[StringTableReaderParameters]
    StringTableReader(fs, params)
  }
}

case class StringTablePartitionReader(lines: GenericLines, uidFieldName: String) extends PartitionReader{
  override def contextType: Type = lines.contextType

  override def fullRowType: TStruct = TStruct("file"-> TString, "text"-> TString, uidFieldName -> TTuple(TInt64, TInt64))

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val req = BaseTypeWithRequiredness(requestedType).asInstanceOf[RStruct]
    req.fields.foreach(field => field.typ.hardSetRequiredness(true))
    req.hardSetRequiredness(true)
    req
  }

  override def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct
  ): IEmitCode = {

    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)))

    context.toI(cb).map(cb) { case partitionContext: SBaseStructValue =>
      val iter = mb.genFieldThisRef[CloseableIterator[GenericLine]]("string_table_reader_iter")

      val fileName = mb.genFieldThisRef[String]("fileName")
      val line = mb.genFieldThisRef[String]("line")
      val partIdx = mb.genFieldThisRef[Long]("partitionIdx")
      val rowIdx = mb.genFieldThisRef[Long]("rowIdx")

      SStreamValue(new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          val contextAsJavaValue = coerce[Any](StringFunctions.svalueToJavaValue(cb, partitionRegion, partitionContext))

          cb.assign(fileName, partitionContext.loadField(cb, "file").get(cb).asString.loadString(cb))
          cb.assign(partIdx, partitionContext.loadField(cb, "partitionIndex").get(cb).asInt.value.toL)
          cb.assign(rowIdx, -1L)

          cb.assign(iter,
            cb.emb.getObject[(FS, Any) => CloseableIterator[GenericLine]](lines.body)
              .invoke[Any, Any, CloseableIterator[GenericLine]]("apply", cb.emb.getFS, contextAsJavaValue)
          )
        }

        override val elementRegion: Settable[Region] =
          mb.genFieldThisRef[Region]("string_table_reader_region")

        override val requiresMemoryManagementPerElement: Boolean = true

        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          val hasNext = iter.invoke[Boolean]("hasNext")
          cb.ifx(hasNext, {
            val gLine = iter.invoke[GenericLine]("next")
            cb.assign(line, gLine.invoke[String]("toString"))
            cb.assign(rowIdx, rowIdx + 1L)
            cb.goto(LproduceElementDone)
          }, {
            cb.goto(LendOfStream)
          })
        }
        override val element: EmitCode = EmitCode.fromI(cb.emb) { cb =>
          val uid = EmitValue.present(
            new SStackStructValue(uidSType, Array(
              EmitValue.present(new SInt64Value(partIdx)),
              EmitValue.present(new SInt64Value(rowIdx)))))
          val requestedFields = IndexedSeq[Option[EmitCode]](
            requestedType.selfField("file").map(_ => EmitCode.present(cb.emb, SJavaString.construct(cb, fileName))),
            requestedType.selfField("text").map(_ => EmitCode.present(cb.emb, SJavaString.construct(cb, line))),
            requestedType.selfField(uidFieldName).map(_ => uid)
          ).flatten.toIndexedSeq
          IEmitCode.present(cb, SStackStruct.constructFromArgs(cb, elementRegion, requestedType,
            requestedFields: _*))
        }

        override def close(cb: EmitCodeBuilder): Unit = {
          cb += iter.invoke[Unit]("close")
        }
      })
    }
  }

  override def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

case class StringTableReader(
  val params: StringTableReaderParameters,
  fileListEntries: IndexedSeq[FileListEntry]
) extends TableReaderWithExtraUID {

  override def uidType = TTuple(TInt64, TInt64)

  override def fullTypeWithoutUIDs: TableType = TableType(
    TStruct("file"-> TString, "text" -> TString),
    FastIndexedSeq.empty,
    TStruct())

  override def renderShort(): String = defaultRender()

  override def pathsUsed: Seq[String] = params.files

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val fs = ctx.fs
    val lines = GenericLines.read(fs, fileListEntries, None, None, params.minPartitions, params.forceBGZ, params.forceGZ,
      params.filePerPartition)
    TableStage(globals = MakeStruct(FastSeq()),
      partitioner = RVDPartitioner.unkeyed(ctx.stateManager, lines.nPartitions),
      dependency = TableStageDependency.none,
      contexts = ToStream(Literal.coerce(TArray(lines.contextType), lines.contexts)),
      body = { partitionContext: Ref => ReadPartition(partitionContext, requestedType.rowType, StringTablePartitionReader(lines, uidFieldName))
      }
    )
  }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    throw new LowererUnsupportedOperation(s"${ getClass.getSimpleName }.lowerGlobals not implemented")

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalStruct(
      IndexedSeq(PField("file", PCanonicalString(true), 0),
        PField("text", PCanonicalString(true), 1)),
      true
    ).subsetTo(requestedType.rowType))

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalStruct.empty(required = true))
}
