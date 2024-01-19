package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Arbitrary._
import is.hail.check.Gen

import scala.reflect.{ClassTag, _}

case object TString extends Type {
  def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit =
    sb.append("str")

  override def _showStr(a: Annotation): String = "\"" + a.asInstanceOf[String] + "\""

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[String]

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = arbitrary[String]

  override def scalaClassTag: ClassTag[String] = classTag[String]

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    ExtendedOrdering.extendToNull(implicitly[Ordering[String]], missingEqual)
}
