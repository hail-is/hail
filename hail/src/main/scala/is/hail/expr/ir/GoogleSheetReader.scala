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
import is.hail.types.physical.{PCanonicalArray, PCanonicalString, PCanonicalStruct, PField, PStruct, PType}
import is.hail.types.virtual.{TArray, TString, TStruct, Type}
import is.hail.types.{BaseTypeWithRequiredness, RStruct, TableType, TypeWithRequiredness}
import is.hail.types.virtual.{Field, TArray, TStream, TString, TStruct, Type}
import is.hail.utils.{FastIndexedSeq, FastSeq, checkGzippedFile, fatal}
import org.json4s.{Extraction, Formats, JValue}

case class GoogleSheetReaderParameters(
                                        spreadsheetID:String,
                                        sheetname:String)

object GoogleSheetReader {
  def apply(fs: FS, params: GoogleSheetReaderParameters): GoogleSheetReader = {
    new GoogleSheetReader(params)
  }
  def fromJValue(fs: FS, jv: JValue): GoogleSheetReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[GoogleSheetReaderParameters]
    GoogleSheetReader(fs, params)
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

class GoogleSheetReader(
                         val params: GoogleSheetReaderParameters ) extends TableReader {

  val fullType: TableType = TableType(
    TStruct("row"-> TArray(TString)),
    FastIndexedSeq.empty,
    TStruct())
  override def renderShort(): String = defaultRender()

  override def pathsUsed: Seq[String] = FastSeq(params.spreadsheetID)

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {null
  }

  override def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val ts = lower(ctx, tr.typ)
    val (broadCastRow, rVD) = TableStageToRVD.apply(ctx, ts, Map[String, IR]())
    TableValue(ctx, tr.typ, broadCastRow, rVD)
  }
  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) =
    (
      PCanonicalStruct(
        IndexedSeq(
          PField(
            "row", PCanonicalArray(PCanonicalString(true)), 0)),
        true).subsetTo(requestedType.rowType).asInstanceOf[PStruct],
      PCanonicalStruct.empty(required = true)
    )
}

