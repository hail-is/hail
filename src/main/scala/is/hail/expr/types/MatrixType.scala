package is.hail.expr.types

import is.hail.expr.EvalContext
import is.hail.rvd.OrderedRVDType
import is.hail.utils._
import is.hail.variant.{GenomeReference, Genotype}

object MatrixType {
  def locusType(vType: Type): Type = {
    vType match {
      case t: TVariant => TLocus(t.gr)
      case _ => vType
    }
  }

  def apply(
    globalType: TStruct = TStruct.empty(),
    colType: TStruct = TStruct.empty(),
    colKey: IndexedSeq[String] = Array.empty[String],
    vType: Type = TVariant(GenomeReference.defaultReference),
    vaType: TStruct = TStruct.empty(),
    genotypeType: TStruct = Genotype.htsGenotypeType): MatrixType = {
    val lType = locusType(vType)

    val vField = vaType.uniqueFieldName("v")
    val pkField = vaType.uniqueFieldName("pk")
    val t = MatrixType(
      globalType,
      colKey,
      colType,
      Array(pkField),
      Array(pkField, vField),
      TStruct(pkField -> lType, vField -> vType) ++ vaType,
      genotypeType)
    assert(t.saType == colType)
    assert(t.vType == vType)
    assert(t.vaType == vaType)
    assert(t.locusType == lType)
    t
  }
}

case class MatrixType(
  globalType: TStruct,
  colKey: IndexedSeq[String],
  colType: TStruct,
  rowPartitionKey: IndexedSeq[String],
  rowKey: IndexedSeq[String],
  rowType: TStruct,
  entryType: TStruct) extends BaseType {
  assert({
    val colFields = colType.fieldNames.toSet
    colKey.forall(colFields.contains)
  }, s"$colKey: $colType")

  assert(rowPartitionKey.length == 1)
  val pkField: String = rowPartitionKey(0)

  assert(rowKey.length == 2)
  assert(rowKey(0) == pkField)
  val vField = rowKey(1)

  val vType: Type = rowType.field(vField).typ

  val locusType: Type = MatrixType.locusType(vType)

  val (vaType, _) = rowType.filter(f => f.name != pkField && f.name != vField)

  val saType = colType

  def genotypeType: TStruct = entryType

  // FIXME needs to be rowType ++ TStruct(entriesField -> TArray(entryType))
  val rvRowType: TStruct =
  TStruct(
    "pk" -> locusType,
    "v" -> vType,
    "va" -> vaType,
    "gs" -> TArray(genotypeType))

  def orderedRVType: OrderedRVDType = {
    new OrderedRVDType(Array("pk"),
      Array("pk", "v"),
      rvRowType)
  }

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, saType),
      "g" -> (2, genotypeType),
      "v" -> (3, vType),
      "va" -> (4, vaType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "sa" -> (1, saType),
      "gs" -> (2, TAggregable(genotypeType, aggregationST))))
  }

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "g" -> (3, genotypeType),
      "sa" -> (4, saType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "gs" -> (3, TAggregable(genotypeType, aggregationST))))
  }

  def genotypeEC: EvalContext = {
    EvalContext(Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "sa" -> (3, saType),
      "g" -> (4, genotypeType)))
  }

  def copy(globalType: TStruct = globalType,
    saType: TStruct = saType,
    colKey: IndexedSeq[String] = colKey,
    vType: Type = vType,
    vaType: TStruct = vaType,
    genotypeType: TStruct = genotypeType): MatrixType =
    MatrixType(globalType, saType, colKey, vType, vaType, genotypeType)

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
}
