package is.hail.expr.ir

import is.hail.types.virtual._

import scala.reflect.{classTag, ClassTag}

object PrimitiveTypeToIRIntermediateClassTag {
  def apply(t: Type): ClassTag[_] = t match {
    case TBoolean => classTag[Boolean]
    case TInt32 => classTag[Int]
    case TInt64 => classTag[Long]
    case TFloat32 => classTag[Float]
    case TFloat64 => classTag[Double]
  }
}
