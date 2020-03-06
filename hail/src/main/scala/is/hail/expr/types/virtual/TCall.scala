package is.hail.expr.types.virtual

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.types._
import is.hail.expr.types.physical.PCall
import is.hail.variant.Call

import scala.reflect.{ClassTag, _}

case object TCall extends ComplexType {
  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("call")
  }
  val representation: Type = TInt32

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genNonmissingValue

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def str(a: Annotation): String = if (a == null) "NA" else Call.toString(a.asInstanceOf[Call])

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])
}
