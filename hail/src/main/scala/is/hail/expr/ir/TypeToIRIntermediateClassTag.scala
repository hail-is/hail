package is.hail.expr.ir

import is.hail.expr.types._

import scala.reflect.{ClassTag, classTag}

object TypeToIRIntermediateClassTag {
  def apply(t: Type): ClassTag[_] = t.fundamentalType match {
    case TVoid => classTag[Unit]
    case _: TBoolean => classTag[Boolean]
    case _: TInt32 => classTag[Int]
    case _: TInt64 => classTag[Long]
    case _: TFloat32 => classTag[Float]
    case _: TFloat64 => classTag[Double]
    case _: TBaseStruct | _: TArray | _: TBinary => classTag[Long]
  }
}
