package is.hail.expr.types

import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.utils._

import scala.reflect.{ClassTag, _}

class TString(override val required: Boolean) extends Type {
  def _toString = "String"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def genNonmissingValue: Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = TBinary(required).unsafeOrdering(missingGreatest)

  def ordering(missingGreatest: Boolean): Ordering[Annotation] =
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(implicitly[Ordering[String]]))

  override def byteSize: Long = 8

  override def fundamentalType: Type = TBinary(required)

}

object TString {
  def apply(required: Boolean = false): TString = if (required) TStringRequired else TStringOptional

  def unapply(t: TString): Option[Boolean] = Option(t.required)

  def loadString(region: Region, boff: Long): String = {
    val length = TBinary.loadLength(region, boff)
    new String(region.loadBytes(TBinary.bytesOffset(boff), length))
  }
}
