package is.hail.expr.types

import is.hail.expr.EvalContext
import is.hail.rvd.OrderedRVType
import is.hail.utils._
import is.hail.variant.{GenomeReference, Genotype}

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

  val entriesIdx = rvRowType.fieldIdx(MatrixType.entriesIdentifier)
  val rowType: TStruct = TStruct(rvRowType.fields.filter(_.index != entriesIdx).map(f => (f.name, f.typ)): _*)
  val entryArrayType: TArray = rvRowType.fieldType(entriesIdx).asInstanceOf[TArray]
  val entryType: TStruct = entryArrayType.elementType.asInstanceOf[TStruct]

  assert({
    val rowFields = rowType.fieldNames.toSet
    rowKey.forall(rowFields.contains)
  }, s"$rowKey: $rowType")

  assert(rowKey.startsWith(rowPartitionKey))

  val rowKeyStruct = TStruct(rowKey.map(k => k -> rowType.fieldByName(k).typ): _*)

  def orderedRVType: OrderedRVType = {
    new OrderedRVType(rowPartitionKey.toArray,
      rowKey.toArray,
      rvRowType)
  }

  def colEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "va" -> (3, rowType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "gs" -> (2, TAggregable(entryType, aggregationST))))
  }

  def rowEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "g" -> (2, entryType),
      "sa" -> (3, colType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "gs" -> (2, TAggregable(entryType, aggregationST))))
  }

  def genotypeEC: EvalContext = {
    EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
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
    colKey.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(",$space"))
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
      warn(s"deleted row partition key [${this.rowPartitionKey.mkString(", ")}], " +
        s"all downstream operations will be single-threaded")
    }

    MatrixType.fromParts(globalType, colKey, colType, rowPartitionKey, rowKey, rowType, entryType)
  }
}
