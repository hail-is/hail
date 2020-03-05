package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual._

import scala.reflect.{ClassTag, classTag}

object TypeToIRIntermediateClassTag {
  def apply(t: Type): ClassTag[_] = t.fundamentalType match {
    case TVoid => classTag[Unit]
    case TBoolean => classTag[Boolean]
    case TInt32 => classTag[Int]
    case TInt64 => classTag[Long]
    case TFloat32 => classTag[Float]
    case TFloat64 => classTag[Double]
    case _: TBaseStruct | _: TArray | TBinary => classTag[Long]
  }
}
