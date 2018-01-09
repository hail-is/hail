package is.hail.expr.types

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.utils._

import scala.reflect.{ClassTag, _}

class TBoolean(override val required: Boolean) extends Type {
  def _toString = "Boolean"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genNonmissingValue: Gen[Annotation] = arbitrary[Boolean]

  override def scalaClassTag: ClassTag[java.lang.Boolean] = classTag[java.lang.Boolean]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Boolean.compare(r1.loadBoolean(o1), r2.loadBoolean(o2))
    }
  }

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[Boolean]]))

  override def byteSize: Long = 1
}

object TBoolean {
  def apply(required: Boolean = false): TBoolean = if (required) TBooleanRequired else TBooleanOptional

  def unapply(t: TBoolean): Option[Boolean] = Option(t.required)
}
