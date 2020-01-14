package is.hail.expr.types.virtual

import is.hail.annotations._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.types.physical.PBoolean

import scala.reflect.{ClassTag, _}

case object TBooleanOptional extends TBoolean(false)
case object TBooleanRequired extends TBoolean(true)

class TBoolean(override val required: Boolean) extends Type {
  def _toPretty = "Boolean"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("bool")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Boolean]

  override def scalaClassTag: ClassTag[java.lang.Boolean] = classTag[java.lang.Boolean]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Boolean]])
}

object TBoolean {
  def apply(required: Boolean = false): TBoolean = if (required) TBooleanRequired else TBooleanOptional

  def unapply(t: TBoolean): Option[Boolean] = Option(t.required)
}
