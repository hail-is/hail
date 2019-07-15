package is.hail.expr.types.encoded

import is.hail.expr.types.physical._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EType {
  def toPType(): PType = PBinary(required)
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}

case object EStringOptional extends EString(false)
case object EStringRequired extends EString(true)

class EString(override val required: Boolean) extends EType {
  def toPType(): PType = PString(required)
}

object EString {
  def apply(required: Boolean = false): EString = if (required) EStringRequired else EStringOptional
}
