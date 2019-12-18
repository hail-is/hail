package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TDict

object PDict {
  def apply(keyType: PType, valueType: PType, required: Boolean = false) = PCanonicalDict(keyType, valueType, required)
}

abstract class PDict extends PContainer {
  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType, required)

  val keyType: PType
  val valueType: PType

  def copy(keyType: PType = this.keyType, valueType: PType = this.valueType, required: Boolean = this.required): PDict

  def elementType: PStruct

  def arrayFundamentalType: PArray = fundamentalType.asInstanceOf[PArray]

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.mapOrdering(this, other.asInstanceOf[PDict], mb)
  }

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("dict<")
    keyType.pyString(sb)
    sb.append(", ")
    valueType.pyString(sb)
    sb.append('>')
  }
}
