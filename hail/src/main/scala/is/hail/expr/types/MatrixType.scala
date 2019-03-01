package is.hail.expr.types

import is.hail.annotations.Annotation
import is.hail.expr.ir.{Env, IRParser}
import is.hail.expr.types.physical.{PArray, PStruct}
import is.hail.expr.types.virtual._
import is.hail.rvd.RVDType
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString


class MatrixTypeSerializer extends CustomSerializer[MatrixType](format => (
  { case JString(s) => IRParser.parseMatrixType(s) },
  { case mt: MatrixType => JString(mt.toString) }))

object MatrixType {
  val entriesIdentifier = "the entries! [877f12a8827e18f61222c6c8c5fb04a8]"

  def getRowType(rvRowType: PStruct): PStruct = rvRowType.dropFields(Set(entriesIdentifier))
  def getEntryArrayType(rvRowType: PStruct): PArray = rvRowType.field(entriesIdentifier).typ.asInstanceOf[PArray]
  def getSplitEntriesType(rvRowType: PStruct): PStruct = rvRowType.selectFields(Array(entriesIdentifier))
  def getEntryType(rvRowType: PStruct): PStruct = getEntryArrayType(rvRowType).elementType.asInstanceOf[PStruct]
  def getEntriesIndex(rvRowType: PStruct): Int = rvRowType.fieldIdx(entriesIdentifier)

  def fromParts(
    globalType: TStruct,
    colKey: IndexedSeq[String],
    colType: TStruct,
    rowKey: IndexedSeq[String],
    rowType: TStruct,
    entryType: TStruct
  ): MatrixType = {
    MatrixType(globalType, colKey, colType, rowKey, rowType ++ TStruct(entriesIdentifier -> TArray(entryType)))
  }
}

case class MatrixType(
  globalType: TStruct,
  colKey: IndexedSeq[String],
  colType: TStruct,
  rowKey: IndexedSeq[String],
  rvRowType: TStruct
) extends BaseType {
  assert({
    val colFields = colType.fieldNames.toSet
    colKey.forall(colFields.contains)
  }, s"$colKey: $colType")

  val entriesIdx: Int = rvRowType.fieldIdx(MatrixType.entriesIdentifier)
  val rowType: TStruct = TStruct(rvRowType.fields.filter(_.index != entriesIdx).map(f => (f.name, f.typ)): _*)
  val entryArrayType: TArray = rvRowType.types(entriesIdx).asInstanceOf[TArray]
  val entryType: TStruct = entryArrayType.elementType.asInstanceOf[TStruct]

  val entriesRVType: TStruct = TStruct(
    MatrixType.entriesIdentifier -> TArray(entryType))

  assert({
    val rowFields = rowType.fieldNames.toSet
    rowKey.forall(rowFields.contains)
  }, s"$rowKey: $rowType")

  val (rowKeyStruct, _) = rowType.select(rowKey)
  def extractRowKey: Row => Row = rowType.select(rowKey)._2
  val rowKeyFieldIdx: Array[Int] = rowKey.toArray.map(rowType.fieldIdx)
  val (rowValueStruct, _) = rowType.filterSet(rowKey.toSet, include = false)
  def extractRowValue: Annotation => Annotation = rowType.filterSet(rowKey.toSet, include = false)._2
  val rowValueFieldIdx: Array[Int] = rowValueStruct.fieldNames.map(rowType.fieldIdx)

  val (colKeyStruct, _) = colType.select(colKey)
  def extractColKey: Row => Row = colType.select(colKey)._2
  val colKeyFieldIdx: Array[Int] = colKey.toArray.map(colType.fieldIdx)
  val (colValueStruct, _) = colType.filterSet(colKey.toSet, include = false)
  def extractColValue: Annotation => Annotation = colType.filterSet(colKey.toSet, include = false)._2
  val colValueFieldIdx: Array[Int] = colValueStruct.fieldNames.map(colType.fieldIdx)

  val colsTableType: TableType =
    TableType(colType, colKey, globalType)

  val rowsTableType: TableType =
    TableType(rowType, rowKey, globalType)

  lazy val entriesTableType: TableType = {
    val resultStruct = TStruct((rowType.fields ++ colType.fields ++ entryType.fields).map(f => f.name -> f.typ): _*)
    TableType(resultStruct, rowKey ++ colKey, globalType)
  }

  def canonicalRVDType: RVDType = RVDType(rvRowType.physicalType, rowKey)

  def refMap: Map[String, Type] = Map(
    "global" -> globalType,
    "va" -> rvRowType,
    "sa" -> colType,
    "g" -> entryType)

  def pretty(sb: StringBuilder, indent0: Int = 0, compact: Boolean = false) {
    var indent = indent0

    val space: String = if (compact) "" else " "

    def newline() {
      if (!compact) {
        sb += '\n'
        sb.append(" " * indent)
      }
    }

    sb.append(s"Matrix$space{")
    indent += 4
    newline()

    sb.append(s"global:$space")
    globalType.pretty(sb, indent, compact)
    sb += ','
    newline()

    sb.append(s"col_key:$space[")
    colKey.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(s",$space"))
    sb += ']'
    sb += ','
    newline()

    sb.append(s"col:$space")
    colType.pretty(sb, indent, compact)
    sb += ','
    newline()

    sb.append(s"row_key:$space[[")
    rowKey.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(s",$space"))
    sb ++= "]]"
    sb += ','
    newline()

    sb.append(s"row:$space")
    rowType.pretty(sb, indent, compact)
    sb += ','
    newline()

    sb.append(s"entry:$space")
    entryType.pretty(sb, indent, compact)

    indent -= 4
    newline()
    sb += '}'
  }

  def copyParts(
    globalType: TStruct = globalType,
    colKey: IndexedSeq[String] = colKey,
    colType: TStruct = colType,
    rowKey: IndexedSeq[String] = rowKey,
    rowType: TStruct = rowType,
    entryType: TStruct = entryType
  ): MatrixType = {
    MatrixType.fromParts(globalType, colKey, colType, rowKey, rowType, entryType)
  }

  @transient lazy val globalEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)

  @transient lazy val rowEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)
    .bind("va" -> rvRowType)

  @transient lazy val colEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)
    .bind("sa" -> colType)

  @transient lazy val entryEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)
    .bind("sa" -> colType)
    .bind("va" -> rvRowType)
    .bind("g" -> entryType)

  def requireRowKeyVariant() {
    val rowKeyTypes = rowKeyStruct.types
    rowKey.zip(rowKeyTypes) match {
      case IndexedSeq(("locus", TLocus(_, _)), ("alleles", TArray(TString(_), _))) =>
    }
  }

  def requireColKeyString() {
    colKeyStruct.types match {
      case Array(_: TString) =>
    }
  }

  def referenceGenome: ReferenceGenome = {
    val firstKeyField = rowKeyStruct.types(0)
    firstKeyField match {
      case TLocus(rg: ReferenceGenome, _) => rg
    }
  }
}
