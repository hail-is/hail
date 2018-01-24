package is.hail.expr.types

import is.hail.expr.EvalContext
import is.hail.rvd.OrderedRVType
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
    sType: Type = TString(),
    saType: TStruct = TStruct.empty(),
    vType: Type = TVariant(GenomeReference.defaultReference),
    vaType: TStruct = TStruct.empty(),
    genotypeType: TStruct = Genotype.htsGenotypeType): MatrixType = {
    val lType = locusType(vType)

    val sField = saType.uniqueFieldName("s")
    val vField = vaType.uniqueFieldName("v")
    val pkField = vaType.uniqueFieldName("pk")
    val t = MatrixType(
      globalType,
      Array(sField),
      TStruct(sField -> sType) ++ saType,
      Array(pkField),
      Array(pkField, vField),
      TStruct(pkField -> lType, vField -> vType) ++ vaType,
      genotypeType)
    assert(t.sType == sType)
    assert(t.saType == saType)
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
  // FIXME rowType
  rowType: TStruct,
  entryType: TStruct) extends BaseType {
  assert(colKey.length == 1)
  val sField: String = colKey(0)

  assert(rowPartitionKey.length == 1)
  val pkField: String = rowPartitionKey(0)

  assert(rowKey.length == 2)
  assert(rowKey(0) == pkField)
  val vField = rowKey(1)

  val vType: Type = rowType.field(vField).typ

  val locusType: Type = MatrixType.locusType(vType)

  val (vaType, _) = rowType.filter(f => f.name != pkField && f.name != vField)

  val sType: Type = colType.field(sField).typ

  val (saType, _) = colType.filter(f => f.name != sField)

  def genotypeType: TStruct = entryType

  // FIXME rename
  // FIXME needs to be rowType ++ TStruct(entriesField -> TArray(entryType))
  val rvRowType: TStruct =
    TStruct(
      "pk" -> locusType,
      "v" -> vType,
      "va" -> vaType,
      "gs" -> TArray(genotypeType))

  def orderedRVType: OrderedRVType = {
    new OrderedRVType(Array("pk"),
      Array("pk", "v"),
      rvRowType)
  }

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "s" -> (1, sType),
      "sa" -> (2, saType),
      "g" -> (3, genotypeType),
      "v" -> (4, vType),
      "va" -> (5, vaType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "s" -> (1, sType),
      "sa" -> (2, saType),
      "gs" -> (3, TAggregable(genotypeType, aggregationST))))
  }

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "v" -> (1, vType),
      "va" -> (2, vaType),
      "g" -> (3, genotypeType),
      "s" -> (4, sType),
      "sa" -> (5, saType))
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
      "s" -> (3, sType),
      "sa" -> (4, saType),
      "g" -> (5, genotypeType)))
  }

  def copy(globalType: TStruct = globalType,
    sType: Type = sType,
    saType: TStruct = saType,
    vType: Type = vType,
    vaType: TStruct = vaType,
    genotypeType: TStruct = genotypeType): MatrixType =
    MatrixType(globalType, sType, saType, vType, vaType, genotypeType)
}