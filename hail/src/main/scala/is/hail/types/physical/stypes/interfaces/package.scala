package is.hail.types.physical.stypes

import is.hail.asm4s._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._

package object interfaces {

  def primitive(x: Code[Long]): SInt64Code = new SInt64Code(x)
  def primitive(x: Code[Int]): SInt32Code = new SInt32Code(x)
  def primitive(x: Code[Double]): SFloat64Code = new SFloat64Code(x)
  def primitive(x: Code[Float]): SFloat32Code = new SFloat32Code(x)
  def primitive(x: Code[Boolean]): SBooleanCode = new SBooleanCode(x)

  def primitive(t: Type, x: Code[_]): SCode = t match {
    case TInt32 => primitive(coerce[Int](x))
    case TInt64 => primitive(coerce[Long](x))
    case TFloat32 => primitive(coerce[Float](x))
    case TFloat64 => primitive(coerce[Double](x))
    case TBoolean => primitive(coerce[Boolean](x))
  }

  def primitive(x: Value[Long]): SInt64Value = new SInt64Value(x)
  def primitive(x: Value[Int]): SInt32Value = new SInt32Value(x)
  def primitive(x: Value[Double]): SFloat64Value = new SFloat64Value(x)
  def primitive(x: Value[Float]): SFloat32Value = new SFloat32Value(x)
  def primitive(x: Value[Boolean]): SBooleanValue = new SBooleanValue(x)

  def primitive(t: Type, x: Value[_]): SValue = t match {
    case TInt32 => primitive(coerce[Int](x))
    case TInt64 => primitive(coerce[Long](x))
    case TFloat32 => primitive(coerce[Float](x))
    case TFloat64 => primitive(coerce[Double](x))
    case TBoolean => primitive(coerce[Boolean](x))
  }
}
