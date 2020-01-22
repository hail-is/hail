package is.hail.expr.types.virtual

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.types._
import is.hail.expr.types.physical.PCall
import is.hail.variant.Call

import scala.reflect.{ClassTag, _}

case object TCallOptional extends TCall(false)
case object TCallRequired extends TCall(true)

class TCall(override val required: Boolean) extends ComplexType {
  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("call")
  }
  val representation: Type = TCall.representation(required)

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genNonmissingValue

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def str(a: Annotation): String = if (a == null) "NA" else Call.toString(a.asInstanceOf[Call])

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])
}

object TCall {
  def apply(required: Boolean = false): TCall = if (required) TCallRequired else TCallOptional

  def unapply(t: TCall): Option[Boolean] = Option(t.required)

  def representation(required: Boolean = false): Type = TInt32(required)
}
