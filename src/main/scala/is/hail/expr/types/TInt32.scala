package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TInt32Optional extends TInt32(false)
case object TInt32Required extends TInt32(true)

class TInt32(override val required: Boolean) extends TIntegral {
  def _toPretty = "Int32"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("int32")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Int]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Int]

  override def scalaClassTag: ClassTag[java.lang.Integer] = classTag[java.lang.Integer]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      Integer.compare(r1.loadInt(o1), r2.loadInt(o2))
    }
  }

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])

  def codeOrdering(mb: EmitMethodBuilder, other: Type): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Int

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Integer, Int, Int, Int]("compare", x, y)

      override def ltNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x < y

      override def lteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x <= y

      override def gtNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x > y

      override def gteqNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x >= y

      override def equivNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Boolean] =
        x.ceq(y)
    }
  }

  override def byteSize: Long = 4
}

object TInt32 {
  def apply(required: Boolean = false) = if (required) TInt32Required else TInt32Optional

  def unapply(t: TInt32): Option[Boolean] = Option(t.required)
}
