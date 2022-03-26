package is.hail.expr.ir
import is.hail.annotations.Region
import is.hail.asm4s.{Code, CodeLabel, Settable, Value}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency, TableStageToRVD}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.{FS, FileStatus}
import is.hail.rvd.RVDPartitioner
import is.hail.types.physical.stypes.concrete.{SJavaString, SStackStruct}
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SStreamValue}
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PField, PStruct}
import is.hail.types.virtual.{TArray, TString, TStruct, Type}
import is.hail.types.{BaseTypeWithRequiredness, RStruct, TableType, TypeWithRequiredness}
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PField, PStruct, PType}
import is.hail.types.virtual.{Field, TArray, TStream, TString, TStruct, Type}
import is.hail.utils.{FastIndexedSeq, FastSeq, checkGzippedFile, fatal}
import org.json4s.{Extraction, Formats, JValue}

case class StringTableReaderParameters(
  files: Array[String],
  minPartitions: Option[Int],
  forceBGZ: Boolean,
  forceGZ: Boolean,
  filePerPartition: Boolean)

object StringTableReader {
  def apply(fs: FS, params: StringTableReaderParameters): StringTableReader = {
     val fileStatuses = getFileStatuses(fs, params.files, params.forceBGZ, params.forceGZ)
    new StringTableReader(params, fileStatuses)
  }
  def fromJValue(fs: FS, jv: JValue): StringTableReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[StringTableReaderParameters]
    StringTableReader(fs, params)
  }

  def getFileStatuses(fs: FS, files: Array[String], forceBGZ: Boolean, forceGZ: Boolean): Array[FileStatus] = {
    val status = fs.globAllStatuses(files)
    if (status.isEmpty)
      fatal(s"arguments refer to no files: ${files.toIndexedSeq}.")
    if (!forceBGZ) {
      status.foreach { status =>
        val file = status.getPath
        if (file.endsWith(".gz"))
          checkGzippedFile(fs, file, forceGZ, forceBGZ)
      }
    }
    status
  }
}
case class StringTablePartitionReader(lines: GenericLines) extends PartitionReader{
  override def contextType: Type = lines.contextType

  override def fullRowType: Type = TStruct("file"-> TString, "text"-> TString)

  override def rowRequiredness(requestedType: Type): TypeWithRequiredness = {
    val req = BaseTypeWithRequiredness.apply(requestedType).asInstanceOf[RStruct]
    req.fields.foreach(field => field.typ.hardSetRequiredness(true))
    req.hardSetRequiredness(true)
    req
  }

  override def emitStream(
     ctx: ExecuteContext,
     cb: EmitCodeBuilder,
     context: EmitCode,
     partitionRegion: Value[Region],
     requestedType: Type): IEmitCode = {

     context.toI(cb).map(cb) { case partitionContext: SBaseStructValue =>
       val iter = cb.emb.genFieldThisRef[CloseableIterator[GenericLine]]("string_table_reader_iter")

       val fileName = cb.emb.genFieldThisRef[String]("fileName")
       val line = cb.emb.genFieldThisRef[String]("line")

       SStreamValue(new StreamProducer {
         override val length: Option[EmitCodeBuilder => Code[Int]] = None

         override def initialize(cb: EmitCodeBuilder): Unit = {
           val contextAsJavaValue = coerce[Any](StringFunctions.svalueToJavaValue(cb, partitionRegion, partitionContext))

           cb.assign(fileName, partitionContext.loadField(cb, "file").get(cb).asString.loadString(cb))

           cb.assign(iter,
             cb.emb.getObject[(FS, Any) => CloseableIterator[GenericLine]](lines.body)
               .invoke[Any, Any, CloseableIterator[GenericLine]]("apply", cb.emb.getFS, contextAsJavaValue)
           )
         }

         override val elementRegion: Settable[Region] =
           cb.emb.genFieldThisRef[Region]("string_table_reader_region")

         override val requiresMemoryManagementPerElement: Boolean = true

         override val LproduceElement: CodeLabel = cb.emb.defineAndImplementLabel { cb =>
           val hasNext = iter.invoke[Boolean]("hasNext")
           cb.ifx(hasNext, {
             val gLine = iter.invoke[GenericLine]("next")
             cb.assign(line, gLine.invoke[String]("toString"))
             cb.goto(LproduceElementDone)
           }, {
             cb.goto(LendOfStream)
           })
         }
         override val element: EmitCode = EmitCode.fromI(cb.emb) { cb =>
           val reqType: TStruct = requestedType.asInstanceOf[TStruct]
           val requestedFields = IndexedSeq[Option[EmitCode]](
             reqType.selfField("file").map(x => EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, SJavaString.construct(cb, fileName)))),
             reqType.selfField("text").map(x => EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, SJavaString.construct(cb, line))))
           ).flatten.toIndexedSeq
           IEmitCode.present(cb, SStackStruct.constructFromArgs(cb, elementRegion, reqType,
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

class StringTableReader(
      val params: StringTableReaderParameters ,
      fileStatuses: IndexedSeq[FileStatus]) extends TableReader {

  val fullType: TableType = TableType(TStruct("file"-> TString, "text" -> TString), FastIndexedSeq.empty, TStruct())
  override def renderShort(): String = defaultRender()

  override def pathsUsed: Seq[String] = params.files

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val fs = ctx.fs
    val lines = GenericLines.read(fs, fileStatuses, None, None, params.minPartitions, false, true,
      params.filePerPartition)
    TableStage(globals = MakeStruct(FastSeq()),
      partitioner = RVDPartitioner.unkeyed(lines.nPartitions),
      dependency = TableStageDependency.none,
      contexts = ToStream(Literal.coerce(TArray(lines.contextType), lines.contexts)),
      body = { partitionContext: Ref => ReadPartition(partitionContext, requestedType.rowType, StringTablePartitionReader(lines))
      }
    )
  }

  override def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val ts = lower(ctx, tr.typ)
    val (broadCastRow, rVD) = TableStageToRVD.apply(ctx, ts, Map[String, IR]())
    TableValue(ctx, tr.typ, broadCastRow, rVD)
  }
  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) =
    (PCanonicalStruct(IndexedSeq(PField("file", PCanonicalString(true), 0),
                                PField("text", PCanonicalString(true), 1)), true).subsetTo(requestedType.rowType).asInstanceOf[PStruct],
     PCanonicalStruct.empty(required = true))
}
