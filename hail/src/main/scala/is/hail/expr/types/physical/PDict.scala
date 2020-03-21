package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TDict

object PDict {
  def apply(keyType: PType, valueType: PType, required: Boolean = false) = PCanonicalDict(keyType, valueType, required)
}

abstract class PDict extends PContainer {
  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType)

  val keyType: PType
  val valueType: PType

  def elementType: PStruct

  def arrayFundamentalType: PArray = fundamentalType.asInstanceOf[PArray]

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.mapOrdering(this, other.asInstanceOf[PDict], mb)
  }

  override def genNonmissingValue: Gen[Annotation] =
    Gen.buildableOf2[Map](Gen.zip(keyType.genValue, valueType.genValue))
}
