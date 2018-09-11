package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PInt64
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TInt64Optional extends TInt64(false)
case object TInt64Required extends TInt64(true)

class TInt64(override val required: Boolean) extends TIntegral {
  def physicalType: PInt64 = PInt64(required)

  def _toPretty = "Int64"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("int64")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Long]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Long]

  override def scalaClassTag: ClassTag[java.lang.Long] = classTag[java.lang.Long]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Long]])

  override def byteSize: Long = 8
}

object TInt64 {
  def apply(required: Boolean = false): TInt64 = if (required) TInt64Required else TInt64Optional

  def unapply(t: TInt64): Option[Boolean] = Option(t.required)
}
