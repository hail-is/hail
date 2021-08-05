package is.hail.expr.ir
import is.hail.annotations.{BroadcastRow, Region, RegionValue, RegionValueBuilder}
import is.hail.asm4s.{Code, CodeLabel, Settable, Value}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.StringTableReader.getFileStatuses
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency, TableStageToRVD}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.{FS, FileStatus}
import is.hail.rvd.RVDPartitioner
import is.hail.types.physical.stypes.concrete.{SJavaString, SStackStruct}
import is.hail.types.physical.stypes.interfaces.SStreamCode
import is.hail.types.{BaseTypeWithRequiredness, RStruct, TableType, TypeWithRequiredness}
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PField, PStruct, PType}
import is.hail.types.virtual.{Field, TArray, TStream, TString, TStruct, Type}
import is.hail.utils.{FastIndexedSeq, FastSeq, fatal}
import org.apache.spark.sql.Row
import org.json4s.{Extraction, Formats, JValue}

case class StringTableReaderParameters(
  files: Array[String],
  minPartitions: Option[Int])

object StringTableReader {
  def apply(fs: FS, params: StringTableReaderParameters): StringTableReader = {
     val fileStatuses = getFileStatuses(fs, params.files)
    new StringTableReader(params, fileStatuses)
  }
  def fromJValue(fs: FS, jv: JValue): StringTableReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[StringTableReaderParameters]
    StringTableReader(fs, params)
  }

  def getFileStatuses(fs: FS, files: Array[String]): Array[FileStatus] = {
    val status = fs.globAllStatuses(files)
    if (status.isEmpty)
      fatal(s"arguments refer to no files: ${files.toIndexedSeq}.")
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

     context.toI(cb).map(cb) { partitionContext =>
       val iter = cb.emb.genFieldThisRef[CloseableIterator[GenericLine]]("iter")

       val ctxMemo = partitionContext.asBaseStruct.memoize(cb, "string_table_reader_ctx")
       val fileName = cb.emb.genFieldThisRef[String]("fileName")
       val line = cb.emb.genFieldThisRef[String]("line")

       SStreamCode(new StreamProducer {
         override val length: Option[EmitCodeBuilder => Code[Int]] = None

         override def initialize(cb: EmitCodeBuilder): Unit = {
           val contextAsJavaValue = coerce[Any](StringFunctions.scodeToJavaValue(cb, partitionRegion, ctxMemo))

           cb.assign(fileName, ctxMemo.loadField(cb, "file").get(cb).asString.loadString())

           cb.assign(iter,
             cb.emb.getObject[(Any) => CloseableIterator[GenericLine]](lines.body)
               .invoke[Any, CloseableIterator[GenericLine]]("apply", contextAsJavaValue)
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

           IEmitCode.present(cb, SStackStruct.constructFromArgs(cb, elementRegion, reqType,
             EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, SJavaString.construct(fileName))),
             EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, SJavaString.construct(line)))))
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
    val lines = GenericLines.read(fs, fileStatuses, None, None, params.minPartitions, false, true)
    TableStage(globals = MakeStruct(FastSeq()),
      partitioner = RVDPartitioner.unkeyed(lines.nPartitions),
      dependency = TableStageDependency.none,
      //TODO figure out if stream requires memory management per element
      contexts = ToStream(Literal.coerce(TArray(lines.contextType), lines.contexts), true),
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

  def executeGeneric(ctx: ExecuteContext): GenericTableValue = {
    val fs = ctx.fs
    val lines = GenericLines.read(fs, fileStatuses,
                                  None, None, params.minPartitions, false, true)
    val partitioner: Option[RVDPartitioner] = None
    val globals: TStruct => Row = _ => Row.empty
    val fullRowType = rowAndGlobalPTypes(ctx, fullType)._1
    val bodyPType: TStruct => PStruct = (requestedRowType: TStruct) => fullRowType.subsetTo(requestedRowType).asInstanceOf[PStruct]
    val linesBody = lines.body
    val body = { (requestedRowType: TStruct) =>
      val requestedPType = bodyPType(requestedRowType)
      val rowFieldNames = requestedRowType.fieldNames

      { (region: Region, context: Any) =>
        val rvb = new RegionValueBuilder(region)
        linesBody(context).map{ bLine =>
          val line = bLine.toString
          rvb.start(requestedPType)
          rvb.startStruct()
          var i = 0
          if (rowFieldNames.contains("file")) rvb.addAnnotation(TString, bLine.file)
          if (rowFieldNames.contains("text")) rvb.addAnnotation(TString, line)
          rvb.endStruct()
          rvb.end()
        }
      }
    }
    new GenericTableValue(partitioner = partitioner,
      fullTableType = fullType,
      globals = globals,
      contextType = lines.contextType,
      contexts = lines.contexts,
      bodyPType = bodyPType,
      body = body)
  }
}
