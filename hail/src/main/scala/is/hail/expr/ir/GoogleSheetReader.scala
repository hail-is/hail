package is.hail.expr.ir

import is.hail.HAIL_PRETTY_VERSION
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency, TableStageToRVD}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.{FS, FileStatus}
import is.hail.rvd.RVDPartitioner
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.stypes.interfaces._
import is.hail.utils._
import org.json4s.{Extraction, Formats, JValue}

import com.google.api.services.sheets.v4.SheetsScopes
import com.google.auth.oauth2.GoogleCredentials
import com.google.auth.http.HttpCredentialsAdapter
import com.google.api.client.auth.oauth2.Credential
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport
import com.google.api.client.json.gson.GsonFactory
import com.google.api.services.sheets.v4.Sheets
import collection.JavaConverters._

class GoogleSheetPartitionIterator(
  private[this] val spreadsheetId: String,
  private[this] val sheetName: String
) {
  private[this] val httpTransport = GoogleNetHttpTransport.newTrustedTransport()
  private[this] val gsonFactory = GsonFactory.getDefaultInstance()
  private[this] val credentials = GoogleCredentials.getApplicationDefault().createScoped(
    List(SheetsScopes.SPREADSHEETS_READONLY).asJava)
  private[this] val sheetsService = new Sheets.Builder(httpTransport, gsonFactory, new HttpCredentialsAdapter(credentials))
    .setApplicationName("Hail " + HAIL_PRETTY_VERSION)
    .build()
  private[this] val theRowsOrNull: java.util.List[java.util.List[Object]] = sheetsService.spreadsheets().values()
    .get(spreadsheetId, sheetName)
    .execute()
    .getValues()
  private[this] val theRowsIterator: Iterator[Array[String]] = if (theRowsOrNull == null) {
    FastIndexedSeq().iterator
  } else {
    theRowsOrNull.asScala.toArray
      .map(_.asScala.toArray.map(_.toString))
      .toFastIndexedSeq.iterator
  }
  private[this] var rowIndex = 0

  def hasNext(): Boolean = {
    rowIndex += 1
    theRowsIterator.hasNext
  }

  def next(): Array[String] = theRowsIterator.next

  def rowIndex(): Int = rowIndex
}

class GoogleSheetPartitionReader(
  private[this] val spreadsheetId: String,
  private[this] val sheetName: String,
  val uidFieldName: String
) extends PartitionReader {
  override def contextType: TStruct = TStruct()

  override def fullRowType: TStruct = TStruct(
    "cells" -> TArray(TString),
    uidFieldName -> TInt64
  )

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val req = BaseTypeWithRequiredness.apply(requestedType).asInstanceOf[RStruct]
    req.field("cells").hardSetRequiredness(true)
    req.field("cells").asInstanceOf[RIterable].elementType.hardSetRequiredness(true)
    req.field(uidFieldName).hardSetRequiredness(true)
    req.hardSetRequiredness(true)
    req
  }

  override def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    context: EmitCode,
    requestedType: Type
  ): IEmitCode = {
    val spreadsheetIdLocal = spreadsheetId
    val sheetNameLocal = sheetName

    context.toI(cb).map(cb) { case partitionContext: SBaseStructValue =>
      val runtimeField = cb.fieldBuilder.newSettable[GoogleSheetPartitionIterator]("googleSheetPartitionIterator")
      val rowField = cb.fieldBuilder.newSettable[Array[String]]("googleSheetPartitionReaderRawRow")
      val rowIdx = cb.emb.genFieldThisRef[Long]("rowIdx")

      SStreamValue(new StreamProducer {
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder): Unit = {
          cb.assign(runtimeField, Code.newInstance[GoogleSheetPartitionIterator, String, String](
            spreadsheetIdLocal,
            sheetNameLocal
          ))
          cb.assign(rowIdx, -1L)
        }

        override val elementRegion: Settable[Region] =
          cb.emb.genFieldThisRef[Region]("google_sheet_partition_reader_region")

        override val requiresMemoryManagementPerElement: Boolean = true

        override val LproduceElement: CodeLabel = cb.emb.defineAndImplementLabel { cb =>
          cb.ifx(runtimeField.invoke[Boolean]("hasNext"), {
            cb.assign(rowField, runtimeField.invoke[Array[String]]("next"))
            cb.assign(rowIdx, rowIdx + const(1L))
            cb.goto(LproduceElementDone)
          }, {
            cb.goto(LendOfStream)
          })
        }

        override val element: EmitCode = EmitCode.fromI(cb.emb) { cb =>
          val reqType: TStruct = requestedType.asInstanceOf[TStruct]
          val requestedFields = Array(
            reqType.selfField("cells").map { _ =>
              EmitCode.fromI(cb.emb) { cb =>
                IEmitCode.present(cb, SJavaArrayString(true).construct(cb, rowField))
              }
            },
            reqType.selfField(uidFieldName).map { _ =>
              EmitCode.fromI(cb.emb) { cb =>
                IEmitCode.present(cb, new SInt64Value(rowIdx))
              }
            }
          ).flatten.toFastIndexedSeq

          IEmitCode.present(
            cb,
            SStackStruct.constructFromArgs(
              cb,
              elementRegion,
              reqType,
              requestedFields: _*
            )
          )
        }

        override def close(cb: EmitCodeBuilder): Unit = {}
      })
    }
  }

  override def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}


case class GoogleSheetReaderParameters(spreadsheetID:String, sheetname:String)

object GoogleSheetReader {
  def apply(fs: FS, params: GoogleSheetReaderParameters): GoogleSheetReader = {
    new GoogleSheetReader(params)
  }

  def fromJValue(fs: FS, jv: JValue): GoogleSheetReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[GoogleSheetReaderParameters]
    GoogleSheetReader(fs, params)
  }
}

class GoogleSheetReader(
  val params: GoogleSheetReaderParameters
) extends TableReaderWithExtraUID {
  override def pathsUsed: Seq[String] = FastSeq(params.spreadsheetID)

  override def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val ts = lower(ctx, tr.typ)
    val (broadCastRow, rVD) = TableStageToRVD.apply(ctx, ts, Map[String, IR]())
    TableValue(ctx, tr.typ, broadCastRow, rVD)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def isDistinctlyKeyed: Boolean = false

  val fullTypeWithoutUIDs: TableType = TableType(
    TStruct("cells"-> TArray(TString)),
    FastIndexedSeq.empty,
    TStruct())

  override def uidType = TInt64

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = VirtualTypeWithReq(
    PCanonicalStruct(
      IndexedSeq(
        PField(
          "cells", PCanonicalArray(PCanonicalString(true)), 0)),
      true).subsetTo(requestedType.rowType).asInstanceOf[PStruct]
  )

  protected def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PInt64Required)

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalStruct.empty(required = true))

  override def renderShort(): String = defaultRender()

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    assert(requestedGlobalsType == TStruct.empty)
    MakeStruct(FastSeq())
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val reader = new GoogleSheetPartitionReader(params.spreadsheetID, params.sheetname, uidFieldName)
    TableStage(
      globals = MakeStruct(FastSeq()),
      partitioner = RVDPartitioner.unkeyed(ctx.stateManager, 1),
      dependency = TableStageDependency.none,
      contexts = ToStream(MakeArray(MakeStruct(FastSeq()))),
      body = { x => ReadPartition(x, requestedType.rowType, reader) }
    )
  }
}

