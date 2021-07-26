package is.hail.expr.ir
import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.StringTableReader.getFileStatuses
import is.hail.expr.ir.lowering.TableStage
import is.hail.io.fs.{FS, FileStatus}
import is.hail.rvd.RVDPartitioner
import is.hail.types.TableType
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PField, PStruct, PType}
import is.hail.types.virtual.{Field, TString, TStruct}
import is.hail.utils.{FastIndexedSeq, fatal}
import org.apache.spark.sql.Row
import org.json4s.{Formats, JValue}

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
      fatal("arguments refer to no files")
    status
  }
}

class StringTableReader(
      val params: StringTableReaderParameters ,
      fileStatuses: IndexedSeq[FileStatus]
      ) extends TableReader {

  val fullType: TableType = TableType(TStruct("file"-> TString, "text" -> TString), FastIndexedSeq.empty, TStruct())
  override def renderShort(): String = defaultRender()

  override def pathsUsed: Seq[String] = params.files

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    executeGeneric(ctx).toTableStage(ctx, requestedType)

  override def apply(tr: TableRead, ctx: ExecuteContext): TableValue =
    executeGeneric(ctx).toTableValue(ctx, tr.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) =
    (PCanonicalStruct(IndexedSeq(PField("file", PCanonicalString(true), 0),
                                PField("text", PCanonicalString(true), 1)), true),
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
      println(s"requestedRowType = ${requestedRowType} req P ${requestedPType}")
      val rowFieldNames = requestedRowType.fieldNames

      { (region: Region, context: Any) =>
        println(s"requestedRowFieldNames: ${rowFieldNames.toIndexedSeq}")
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
