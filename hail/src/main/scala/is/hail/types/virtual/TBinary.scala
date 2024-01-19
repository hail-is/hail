package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.check.Arbitrary._
import is.hail.check.Gen

import scala.reflect.{ClassTag, _}

case object TBinary extends Type {
  def _toPretty = "Binary"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] =
    Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  def mkOrdering(sm: HailStateManager, _missingEqual: Boolean = true): ExtendedOrdering =
    ExtendedOrdering.iterableOrdering(new ExtendedOrdering {
      val missingEqual = _missingEqual

      override def compareNonnull(x: Any, y: Any): Int =
        java.lang.Integer.compare(
          java.lang.Byte.toUnsignedInt(x.asInstanceOf[Byte]),
          java.lang.Byte.toUnsignedInt(y.asInstanceOf[Byte]),
        )
    })
}
