package is.hail.expr.types.encoded

case object EBooleanOptional extends EBoolean(false)
case object EBooleanRequired extends EBoolean(true)

class EBoolean(override val required: Boolean) extends EType

object EBoolean {
  def apply(required: Boolean = false): EBoolean = if (required) EBooleanRequired else EBooleanOptional
}

case object EFloat32Optional extends EFloat32(false)
case object EFloat32Required extends EFloat32(true)

class EFloat32(override val required: Boolean) extends EType

object EFloat32 {
  def apply(required: Boolean = false): EFloat32 = if (required) EFloat32Required else EFloat32Optional
}

case object EFloat64Optional extends EFloat64(false)
case object EFloat64Required extends EFloat64(true)

class EFloat64(override val required: Boolean) extends EType

object EFloat64 {
  def apply(required: Boolean = false): EFloat64 = if (required) EFloat64Required else EFloat64Optional
}

case object EInt32Optional extends EInt32(false)
case object EInt32Required extends EInt32(true)

class EInt32(override val required: Boolean) extends EType

object EInt32 {
  def apply(required: Boolean = false) = if (required) EInt32Required else EInt32Optional
}

case object EInt64Optional extends EInt64(false)
case object EInt64Required extends EInt64(true)

class EInt64(override val required: Boolean) extends EType

object EInt64 {
  def apply(required: Boolean = false): EInt64 = if (required) EInt64Required else EInt64Optional
}
