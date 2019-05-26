package is.hail.annotations

import is.hail.expr.types.virtual._

import scala.reflect.{ClassTag, classTag}

object RegionValueToScala {
  def classTagHail(t: Type): ClassTag[_] = t match {
    case TBoolean(true) => classTag[Boolean]
    case TBoolean(false) => classTag[java.lang.Boolean]
    case TInt32(true) => classTag[Int]
    case TInt32(false) => classTag[java.lang.Integer]
    case TInt64(true) => classTag[Long]
    case TInt64(false) => classTag[java.lang.Long]
    case TFloat32(true) => classTag[Float]
    case TFloat32(false) => classTag[java.lang.Float]
    case TFloat64(true) => classTag[Double]
    case TFloat64(false) => classTag[java.lang.Double]
    case t => throw new RuntimeException(s"classTagHail does not handle $t")
  }
}
