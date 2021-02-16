package is.hail.types.virtual

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.types._
import is.hail.types.physical.PCall
import is.hail.variant.Call

import scala.reflect.{ClassTag, _}

case object TCall extends Type {
  def _toPretty = "Call"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("call")
  }
  val representation: Type = TInt32

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = Call.genNonmissingValue

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def str(a: Annotation): String = if (a == null) "NA" else Call.toString(a.asInstanceOf[Call])

  override val ordering: ExtendedOrdering = mkOrdering()

  override def mkOrdering(missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]], missingEqual)
}
