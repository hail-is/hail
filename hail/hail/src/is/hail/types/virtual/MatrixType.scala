package is.hail.types.virtual

import is.hail.annotations.Annotation
import is.hail.expr.ir.{Env, IRParser, LowerMatrixIR, MatrixIR, Name}
import is.hail.types.physical.{PArray, PStruct}
import is.hail.utils._

import org.apache.spark.sql.Row
import org.json4s.{CustomSerializer, JValue}
import org.json4s.JsonAST.{JArray, JObject, JString}

class MatrixTypeSerializer extends CustomSerializer[MatrixType](format =>
      (
        { case JString(s) => IRParser.parseMatrixType(s) },
        { case mt: MatrixType => JString(mt.toString) },
      )
    )

object MatrixType {
  val entriesIdentifier = "the entries! [877f12a8827e18f61222c6c8c5fb04a8]"

  def getRowType(rvRowType: PStruct): PStruct = rvRowType.dropFields(Set(entriesIdentifier))

  def getEntryArrayType(rvRowType: PStruct): PArray =
    rvRowType.field(entriesIdentifier).typ.asInstanceOf[PArray]

  def getSplitEntriesType(rvRowType: PStruct): PStruct =
    rvRowType.selectFields(Array(entriesIdentifier))

  def getEntryType(rvRowType: PStruct): PStruct =
    getEntryArrayType(rvRowType).elementType.asInstanceOf[PStruct]

  def getEntriesIndex(rvRowType: PStruct): Int = rvRowType.fieldIdx(entriesIdentifier)

  def globalBindings: IndexedSeq[Int] = FastSeq(0)
  def rowInRowBindings: IndexedSeq[Int] = FastSeq(0, 1)
  def colInColBindings: IndexedSeq[Int] = FastSeq(0, 1)
  def rowInEntryBindings: IndexedSeq[Int] = FastSeq(0, 2)
  def colInEntryBindings: IndexedSeq[Int] = FastSeq(0, 1)
  def entryBindings: IndexedSeq[Int] = FastSeq(0, 1, 2, 3)

  def fromTableType(
    typ: TableType,
    colsFieldName: String,
    entriesFieldName: String,
    colKey: IndexedSeq[String],
  ): MatrixType = {

    val (colType, colsFieldIdx) = typ.globalType.field(colsFieldName) match {
      case Field(_, TArray(t @ TStruct(_)), idx) => (t, idx)
      case Field(_, t, _) => fatal(s"expected cols field to be an array of structs, found $t")
    }
    val newRowType = typ.rowType.deleteKey(entriesFieldName)
    val entryType =
      typ.rowType.field(entriesFieldName).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]

    MatrixType(
      typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
      colKey,
      colType,
      typ.key,
      newRowType,
      entryType,
    )
  }
}

case class MatrixType(
  globalType: TStruct,
  colKey: IndexedSeq[String],
  colType: TStruct,
  rowKey: IndexedSeq[String],
  rowType: TStruct,
  entryType: TStruct,
) extends VType {
  assert(
    {
      val colFields = colType.fieldNames.toSet
      colKey.forall(colFields.contains)
    },
    s"$colKey: $colType",
  )

  lazy val entriesRVType: TStruct = TStruct(
    MatrixType.entriesIdentifier -> TArray(entryType)
  )

  assert(
    {
      val rowFields = rowType.fieldNames.toSet
      rowKey.forall(rowFields.contains)
    },
    s"$rowKey: $rowType",
  )

  lazy val (rowKeyStruct, _) = rowType.select(rowKey)
  def extractRowKey: Row => Row = rowType.select(rowKey)._2
  lazy val rowKeyFieldIdx: Array[Int] = rowKey.toArray.map(rowType.fieldIdx)
  lazy val (rowValueStruct, _) = rowType.filterSet(rowKey.toSet, include = false)

  def extractRowValue: Annotation => Annotation =
    rowType.filterSet(rowKey.toSet, include = false)._2

  lazy val rowValueFieldIdx: Array[Int] = rowValueStruct.fieldNames.map(rowType.fieldIdx)

  lazy val (colKeyStruct, _) = colType.select(colKey)
  def extractColKey: Row => Row = colType.select(colKey)._2
  lazy val colKeyFieldIdx: Array[Int] = colKey.toArray.map(colType.fieldIdx)
  lazy val (colValueStruct, _) = colType.filterSet(colKey.toSet, include = false)

  def extractColValue: Annotation => Annotation =
    colType.filterSet(colKey.toSet, include = false)._2

  lazy val colValueFieldIdx: Array[Int] = colValueStruct.fieldNames.map(colType.fieldIdx)

  lazy val colsTableType: TableType =
    TableType(colType, colKey, globalType)

  lazy val rowsTableType: TableType =
    TableType(rowType, rowKey, globalType)

  lazy val entriesTableType: TableType = {
    val resultStruct =
      TStruct((rowType.fields ++ colType.fields ++ entryType.fields).map(f => f.name -> f.typ): _*)
    TableType(resultStruct, rowKey ++ colKey, globalType)
  }

  lazy val canonicalTableType: TableType =
    toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)

  def toTableType(entriesFieldName: String, colsFieldName: String): TableType = TableType(
    rowType = rowType.appendKey(entriesFieldName, TArray(entryType)),
    key = rowKey,
    globalType = globalType.appendKey(colsFieldName, TArray(colType)),
  )

  def isCompatibleWith(tt: TableType): Boolean = {
    val globalType2 = tt.globalType.deleteKey(LowerMatrixIR.colsFieldName)
    val colType2 =
      tt.globalType.field(LowerMatrixIR.colsFieldName).typ.asInstanceOf[TArray].elementType
    val rowType2 = tt.rowType.deleteKey(LowerMatrixIR.entriesFieldName)
    val entryType2 =
      tt.rowType.field(LowerMatrixIR.entriesFieldName).typ.asInstanceOf[TArray].elementType

    globalType == globalType2 && colType == colType2 && rowType == rowType2 && entryType == entryType2 && rowKey == tt.key
  }

  def isCanonical: Boolean = rowType.isCanonical && globalType.isCanonical && colType.isCanonical

  def refMap: Map[String, Type] = Map(
    "global" -> globalType,
    "va" -> rowType,
    "sa" -> colType,
    "g" -> entryType,
  )

  override def pretty(sb: StringBuilder, indent0: Int = 0, compact: Boolean = false): Unit = {

    val space: String = if (compact) "" else " "
    val newline: String = if (compact) "" else "\n"
    val padding: String = if (compact) "" else "\n" + (" " * indent0)

    val indent = indent0 + 4

    sb ++= "Matrix" ++= space += '{' ++= newline: Unit

    sb ++= padding ++= "global:" ++= space: Unit
    globalType.pretty(sb, indent, compact)
    sb += ',' ++= newline: Unit

    sb ++= padding ++= "col_key:" ++= space += '[': Unit
    colKey.foreachBetween(k => sb ++= prettyIdentifier(k))(sb += ',' ++= space: Unit)
    sb ++= "]," ++= newline: Unit

    sb ++= padding ++= "col:" ++= space: Unit
    colType.pretty(sb, indent, compact)
    sb += ',' ++= newline: Unit

    sb ++= padding ++= "row_key:" ++= space ++= "[[": Unit
    rowKey.foreachBetween(k => sb ++= prettyIdentifier(k))(sb += ',' ++= space: Unit)
    sb ++= "]]," ++= newline: Unit

    sb ++= padding ++= "row:" ++= space: Unit
    rowType.pretty(sb, indent, compact)
    sb += ',' ++= newline: Unit

    sb ++= padding ++= "entry:" ++= space: Unit
    entryType.pretty(sb, indent, compact)
    sb ++= newline
    sb += '}'
  }

  @transient lazy val globalEnv: Env[Type] = Env.empty[Type]
    .bind(globalBindings: _*)

  def globalBindings: IndexedSeq[(Name, Type)] =
    FastSeq(MatrixIR.globalName -> globalType)

  @transient lazy val rowEnv: Env[Type] = Env.empty[Type]
    .bind(rowBindings: _*)

  def rowBindings: IndexedSeq[(Name, Type)] =
    FastSeq(MatrixIR.globalName -> globalType, MatrixIR.rowName -> rowType)

  @transient lazy val colEnv: Env[Type] = Env.empty[Type]
    .bind(colBindings: _*)

  def colBindings: IndexedSeq[(Name, Type)] =
    FastSeq(MatrixIR.globalName -> globalType, MatrixIR.colName -> colType)

  @transient lazy val entryEnv: Env[Type] = Env.empty[Type]
    .bind(entryBindings: _*)

  def entryBindings: IndexedSeq[(Name, Type)] =
    FastSeq(
      MatrixIR.globalName -> globalType,
      MatrixIR.colName -> colType,
      MatrixIR.rowName -> rowType,
      MatrixIR.entryName -> entryType,
    )

  def requireRowKeyVariant(): Unit = {
    val rowKeyTypes = rowKeyStruct.types
    rowKey.zip(rowKeyTypes) match {
      case Seq(("locus", TLocus(_)), ("alleles", TArray(TString))) =>
    }
  }

  def requireColKeyString(): Unit =
    colKeyStruct.types match {
      case Array(TString) =>
    }

  def referenceGenomeName: String = {
    val firstKeyField = rowKeyStruct.types(0)
    firstKeyField.asInstanceOf[TLocus].rg
  }

  override def toJSON: JValue =
    JObject(
      "row_type" -> JString(rowType.toString),
      "row_key" -> JArray(rowKey.toList.map(JString)),
      "col_type" -> JString(colType.toString),
      "col_key" -> JArray(colKey.toList.map(JString)),
      "entry_type" -> JString(entryType.toString),
      "global_type" -> JString(globalType.toString),
    )
}
