package is.hail.expr.types

import is.hail.annotations.Annotation
import is.hail.expr.{EvalContext, Parser}
import is.hail.rvd.OrderedRVDType
import is.hail.utils._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString
import org.apache.spark.sql.Row


class MatrixTypeSerializer extends CustomSerializer[MatrixType](format => (
  { case JString(s) => Parser.parseMatrixType(s) },
  { case mt: MatrixType => JString(mt.toString) }))

object MatrixType {
  val entriesIdentifier = "the entries! [877f12a8827e18f61222c6c8c5fb04a8]"

  def fromParts(globalType: TStruct,
    colKey: IndexedSeq[String],
    colType: TStruct,
    rowPartitionKey: IndexedSeq[String],
    rowKey: IndexedSeq[String],
    rowType: TStruct,
    entryType: TStruct): MatrixType = {
    MatrixType(globalType, colKey,
      colType, rowPartitionKey, rowKey, rowType ++ TStruct(entriesIdentifier -> TArray(entryType)))
  }
}

case class MatrixType(
  globalType: TStruct,
  colKey: IndexedSeq[String],
  colType: TStruct,
  rowPartitionKey: IndexedSeq[String],
  rowKey: IndexedSeq[String],
  rvRowType: TStruct) extends BaseType {
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

  assert(rowKey.startsWith(rowPartitionKey))

  val (rowKeyStruct, _) = rowType.select(rowKey.toArray)
  def extractRowKey: Row => Row = rowType.select(rowKey.toArray)._2
  val rowKeyFieldIdx: Array[Int] = rowKey.toArray.map(rowType.fieldIdx)
  val (rowValueStruct, _) = rowType.filterSet(rowKey.toSet, include = false)
  def extractRowValue: Annotation => Annotation = rowType.filterSet(rowKey.toSet, include = false)._2
  val rowValueFieldIdx: Array[Int] = rowValueStruct.fieldNames.map(rowType.fieldIdx)

  val (colKeyStruct, _) = colType.select(colKey.toArray)
  def extractColKey: Row => Row = colType.select(colKey.toArray)._2
  val colKeyFieldIdx: Array[Int] = colKey.toArray.map(colType.fieldIdx)
  val (colValueStruct, _) = colType.filterSet(colKey.toSet, include = false)
  def extractColValue: Annotation => Annotation = colType.filterSet(colKey.toSet, include = false)._2
  val colValueFieldIdx: Array[Int] = colValueStruct.fieldNames.map(colType.fieldIdx)

  val colsTableType: TableType =
    TableType(
      colType,
      Some(colKey),
      globalType)

  val rowsTableType: TableType =
    TableType(
      rowType,
      Some(rowKey),
      globalType)

  lazy val entriesTableType: TableType = {
    val resultStruct = TStruct((rowType.fields ++ colType.fields ++ entryType.fields).map(f => f.name -> f.typ): _*)
    TableType(resultStruct, Some(rowKey ++ colKey), globalType)
  }

  def orvdType: OrderedRVDType = {
    new OrderedRVDType(rowPartitionKey.toArray,
      rowKey.toArray,
      rvRowType)
  }

  def rowORVDType: OrderedRVDType = new OrderedRVDType(rowPartitionKey.toArray, rowKey.toArray, rowType)

  def colEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "va" -> (3, rvRowType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "AGG" -> (2, TAggregable(entryType, aggregationST))))
  }

  def rowEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "va" -> (1, rvRowType),
      "g" -> (2, entryType),
      "sa" -> (3, colType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rvRowType),
      "AGG" -> (2, TAggregable(entryType, aggregationST))))
  }

  def genotypeEC: EvalContext = {
    EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rvRowType),
      "sa" -> (2, colType),
      "g" -> (3, entryType)))
  }

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
    rowPartitionKey.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(s",$space"))
    sb += ']'
    val rowRestKey = rowKey.drop(rowPartitionKey.length)
    if (rowRestKey.nonEmpty) {
      sb += ','
      rowRestKey.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(s",$space"))
    }
    sb += ']'
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

  def copyParts(globalType: TStruct = globalType,
    colKey: IndexedSeq[String] = colKey,
    colType: TStruct = colType,
    rowPartitionKey: IndexedSeq[String] = rowPartitionKey,
    rowKey: IndexedSeq[String] = rowKey,
    rowType: TStruct = rowType,
    entryType: TStruct = entryType): MatrixType = {
    if (this.rowPartitionKey.nonEmpty && rowPartitionKey.isEmpty) {
      warn(s"deleted row partition key [${ this.rowPartitionKey.mkString(", ") }], " +
        s"all downstream operations will be single-threaded")
    }

    MatrixType.fromParts(globalType, colKey, colType, rowPartitionKey, rowKey, rowType, entryType)
  }
}
