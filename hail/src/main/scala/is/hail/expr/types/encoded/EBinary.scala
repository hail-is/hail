package is.hail.expr.types.encoded

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EType

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}

case object EStringOptional extends EString(false)
case object EStringRequired extends EString(true)

class EString(override val required: Boolean) extends EType

object EString {
  def apply(required: Boolean = false): EString = if (required) EStringRequired else EStringOptional
}
