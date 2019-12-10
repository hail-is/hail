package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TDict

object PDict {
  def apply(keyType: PType, valueType: PType, required: Boolean = false) = PCanonicalDict(keyType, valueType, required)
}

abstract class PDict extends PContainer with PArrayBackedContainer {
  val keyType: PType
  val valueType: PType

  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType, required)

    // TODO: FIX
//  override val elementType: PStruct
//  override val fundamentalType: PArray = ???

  def copy(keyType: PType = this.keyType, valueType: PType = this.valueType, required: Boolean = this.required): PDict

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.mapOrdering(this, other.asInstanceOf[PDict], mb)
  }
}
