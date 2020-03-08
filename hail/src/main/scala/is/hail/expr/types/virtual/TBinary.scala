package is.hail.expr.types.virtual

import is.hail.annotations._
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.types.physical.PBinary

import scala.reflect.{ClassTag, _}

case object TBinary extends Type {
  def _toPretty = "Binary"

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Array[Byte]]

  override def genNonmissingValue: Gen[Annotation] = Gen.buildableOf(arbitrary[Byte])

  override def scalaClassTag: ClassTag[Array[Byte]] = classTag[Array[Byte]]

  val ordering: ExtendedOrdering = ExtendedOrdering.iterableOrdering(new ExtendedOrdering {
    override def compareNonnull(x: Any, y: Any): Int =
      java.lang.Integer.compare(
        java.lang.Byte.toUnsignedInt(x.asInstanceOf[Byte]),
        java.lang.Byte.toUnsignedInt(y.asInstanceOf[Byte]))
  })
}
