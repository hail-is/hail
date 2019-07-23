package is.hail.expr.types.virtual

import is.hail.annotations.{Region, _}
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PInt32

import scala.reflect.{ClassTag, _}

case object TInt32Optional extends TInt32(false)
case object TInt32Required extends TInt32(true)

class TInt32(override val required: Boolean) extends TIntegral {
  lazy val physicalType: PInt32 = PInt32(required)

  def _toPretty = "Int32"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("int32")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Int]

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])
}

object TInt32 {
  def apply(required: Boolean = false) = if (required) TInt32Required else TInt32Optional

  def unapply(t: TInt32): Option[Boolean] = Option(t.required)
}
