package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.{TBinary, TBoolean}
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object PBooleanOptional extends PBoolean(false)
case object PBooleanRequired extends PBoolean(true)

class PBoolean(override val required: Boolean) extends PType {
  def virtualType: TBoolean = TBoolean(required)

  def _toPretty = "Boolean"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("bool")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Boolean]

  override def scalaClassTag: ClassTag[java.lang.Boolean] = classTag[java.lang.Boolean]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Boolean.compare(r1.loadBoolean(o1), r2.loadBoolean(o2))
    }
  }

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Boolean]])

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Boolean

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Boolean, Boolean, Boolean, Int]("compare", x, y)
    }
  }

  override def byteSize: Long = 1
}

object PBoolean {
  def apply(required: Boolean = false): PBoolean = if (required) PBooleanRequired else PBooleanOptional

  def unapply(t: PBoolean): Option[Boolean] = Option(t.required)
}
