package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TDict

object PDict {
  def apply(keyType: PType, valueType: PType, required: Boolean): PDict = PDict(new PDictStruct(keyType, valueType), required)
  def apply(keyType: PType, valueType: PType): PDict = PDict(keyType, valueType, false)
}

class PDictStruct(val keyType: PType, val valueType: PType) {
  val structRep: PStruct = PStruct(required = true, "key" -> keyType, "value" -> valueType)
}

final case class PDict(dictType: PDictStruct, required: Boolean = false) extends PArrayBackedContainer(PCanonicalArray(dictType.structRep, required)) {
  val elementType = dictType.structRep
  val keyType = dictType.keyType
  val valueType = dictType.valueType

  lazy val virtualType: TDict = TDict(keyType.virtualType, valueType.virtualType, required)

  override val fundamentalType: PArray = PCanonicalArray(elementType.fundamentalType, required)

  def _asIdent = s"dict_of_${keyType.asIdent}AND${valueType.asIdent}"
  def _toPretty = s"Dict[$keyType, $valueType]"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("dict<")
    keyType.pyString(sb)
    sb.append(", ")
    valueType.pyString(sb)
    sb.append('>')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Dict[")
    keyType.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb.append(", ")
    valueType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    CodeOrdering.mapOrdering(this, other.asInstanceOf[PDict], mb)
  }
}
