package is.hail.expr.types

import is.hail.annotations.Annotation
import is.hail.expr.ir.{ColSym, EntriesSym, EntrySym, Env, GlobalSym, IRParser, Identifier, RowSym, Sym}
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
  def getRowType(rvRowType: PStruct): PStruct = rvRowType.dropFields(Set(EntriesSym))
  def getEntryArrayType(rvRowType: PStruct): PArray = rvRowType.field(EntriesSym).typ.asInstanceOf[PArray]
  def getSplitEntriesType(rvRowType: PStruct): PStruct = rvRowType.selectFields(Array(EntriesSym))
  def getEntryType(rvRowType: PStruct): PStruct = getEntryArrayType(rvRowType).elementType.asInstanceOf[PStruct]
  def getEntriesIndex(rvRowType: PStruct): Int = rvRowType.fieldIdx(EntriesSym)

  def fromParts(
    globalType: TStruct,
    colKey: IndexedSeq[Sym],
    colType: TStruct,
    rowKey: IndexedSeq[Sym],
    rowType: TStruct,
    entryType: TStruct
  ): MatrixType = {
    MatrixType(globalType, colKey, colType, rowKey, rowType ++ TStruct(EntriesSym -> TArray(entryType)))
  }
}

case class MatrixType(
  globalType: TStruct,
  colKey: IndexedSeq[Sym],
  colType: TStruct,
  rowKey: IndexedSeq[Sym],
  rvRowType: TStruct
) extends BaseType {
  assert({
    val colFields = colType.fieldNames.toSet
    colKey.forall(colFields.contains)
  }, s"$colKey: $colType")

  val entriesIdx: Int = rvRowType.fieldIdx(EntriesSym)
  val rowType: TStruct = TStruct(rvRowType.fields.filter(_.index != entriesIdx).map(f => (f.name, f.typ)): _*)
  val entryArrayType: TArray = rvRowType.types(entriesIdx).asInstanceOf[TArray]
  val entryType: TStruct = entryArrayType.elementType.asInstanceOf[TStruct]

  val entriesRVType: TStruct = TStruct(
    EntriesSym -> TArray(entryType))

  assert({
    val rowFields = rowType.fieldNames.toSet
    rowKey.forall(rowFields.contains)
  }, s"$rowKey: $rowType")

  val (rowKeyStruct, _) = rowType.select(rowKey)
  def extractRowKey: Row => Row = rowType.select(rowKey)._2
  val rowKeyFieldIdx: Array[Int] = rowKey.toArray.map(rowType.fieldIdx)
  val (rowValueStruct, _) = rowType.filterSet(rowKey.toSet, include = false)
  def extractRowValue: Annotation => Annotation = rowType.filterSet(rowKey.toSet, include = false)._2
  val rowValueFieldIdx: IndexedSeq[Int] = rowValueStruct.fieldNames.map(rowType.fieldIdx)

  val (colKeyStruct, _) = colType.select(colKey)
  def extractColKey: Row => Row = colType.select(colKey)._2
  val colKeyFieldIdx: Array[Int] = colKey.toArray.map(colType.fieldIdx)
  val (colValueStruct, _) = colType.filterSet(colKey.toSet, include = false)
  def extractColValue: Annotation => Annotation = colType.filterSet(colKey.toSet, include = false)._2
  val colValueFieldIdx: IndexedSeq[Int] = colValueStruct.fieldNames.map(colType.fieldIdx)

  val colsTableType: TableType =
    TableType(colType, colKey, globalType)

  val rowsTableType: TableType =
    TableType(rowType, rowKey, globalType)

  lazy val entriesTableType: TableType = {
    val resultStruct = TStruct((rowType.fields ++ colType.fields ++ entryType.fields).map(f => f.name -> f.typ): _*)
    TableType(resultStruct, rowKey ++ colKey, globalType)
  }

  def canonicalRVDType: RVDType = RVDType(rvRowType.physicalType, rowKey)

  def refMap: Map[Sym, Type] = Map(
    GlobalSym -> globalType,
    RowSym -> rvRowType,
    ColSym -> colType,
    EntrySym -> entryType)

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
    colKey.foreachBetween(k => sb.append(k))(sb.append(s",$space"))
    sb += ']'
    sb += ','
    newline()

    sb.append(s"col:$space")
    colType.pretty(sb, indent, compact)
    sb += ','
    newline()

    sb.append(s"row_key:$space[[")
    rowKey.foreachBetween(k => sb.append(k))(sb.append(s",$space"))
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
    colKey: IndexedSeq[Sym] = colKey,
    colType: TStruct = colType,
    rowKey: IndexedSeq[Sym] = rowKey,
    rowType: TStruct = rowType,
    entryType: TStruct = entryType
  ): MatrixType = {
    MatrixType.fromParts(globalType, colKey, colType, rowKey, rowType, entryType)
  }

  def globalEnv: Env[Type] = Env.empty[Type]
    .bind(GlobalSym -> globalType)

  def rowEnv: Env[Type] = Env.empty[Type]
    .bind(GlobalSym -> globalType)
    .bind(RowSym -> rvRowType)

  def colEnv: Env[Type] = Env.empty[Type]
    .bind(GlobalSym -> globalType)
    .bind(ColSym -> colType)

  def entryEnv: Env[Type] = Env.empty[Type]
    .bind(GlobalSym -> globalType)
    .bind(ColSym -> colType)
    .bind(RowSym -> rvRowType)
    .bind(EntrySym -> entryType)

  def requireRowKeyVariant() {
    val rowKeyTypes = rowKeyStruct.types
    rowKey.zip(rowKeyTypes) match {
      case IndexedSeq((Identifier("locus"), TLocus(_, _)), (Identifier("alleles"), TArray(TString(_), _))) =>
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
