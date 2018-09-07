package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.Code
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PBoolean
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object TBooleanOptional extends TBoolean(false)
case object TBooleanRequired extends TBoolean(true)

class TBoolean(override val required: Boolean) extends Type {
  def physicalType: PBoolean = PBoolean(required)

  def _toPretty = "Boolean"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("bool")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Boolean]

  override def scalaClassTag: ClassTag[java.lang.Boolean] = classTag[java.lang.Boolean]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[Boolean]])

  def codeOrdering(mb: EmitMethodBuilder, other: Type): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      type T = Boolean

      def compareNonnull(rx: Code[Region], x: Code[T], ry: Code[Region], y: Code[T], missingGreatest: Boolean): Code[Int] =
        Code.invokeStatic[java.lang.Boolean, Boolean, Boolean, Int]("compare", x, y)
    }
  }

  override def byteSize: Long = 1
}

object TBoolean {
  def apply(required: Boolean = false): TBoolean = if (required) TBooleanRequired else TBooleanOptional

  def unapply(t: TBoolean): Option[Boolean] = Option(t.required)
}
