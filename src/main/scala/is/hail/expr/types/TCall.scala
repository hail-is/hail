package is.hail.expr.types

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.utils._
import is.hail.variant.Call

import scala.reflect.{ClassTag, _}

case object TCallOptional extends TCall(false)
case object TCallRequired extends TCall(true)

class TCall(override val required: Boolean) extends ComplexType {
  def _toString = "Call"

  val representation: Type = TCall.representation(required)

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genNonmissingValue

  override def desc: String = "A ``Call`` is a Hail data type representing a genotype call (ex: 0/0) in the Variant Dataset."

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
