package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.TCall
import is.hail.variant.Call

import scala.reflect.{ClassTag, _}

case object PCallOptional extends PCall(false)
case object PCallRequired extends PCall(true)

class PCall(override val required: Boolean) extends ComplexPType {
  def virtualType: TCall = TCall(required)

  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("call")
  }
  val representation: PType = PCall.representation(required)

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

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
