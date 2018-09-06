package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._
import is.hail.variant.Call

import scala.reflect.{ClassTag, _}

case object PCallOptional extends PCall(false)
case object PCallRequired extends PCall(true)

class PCall(override val required: Boolean) extends ComplexPType {
  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("call")
  }
  val representation: PType = PCall.representation(required)

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genNonmissingValue

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def str(a: Annotation): String = if (a == null) "NA" else Call.toString(a.asInstanceOf[Call])

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    PInt32().codeOrdering(mb)
  }
}

object PCall {
  def apply(required: Boolean = false): PCall = if (required) PCallRequired else PCallOptional

  def unapply(t: PCall): Option[Boolean] = Option(t.required)

  def representation(required: Boolean = false): PType = PInt32(required)
}
