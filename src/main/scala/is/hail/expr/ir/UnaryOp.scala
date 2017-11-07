package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr._

object UnaryOp {
  def inferReturnType(op: UnaryOp, t: Type): Type = op match {
    case Negate() => t match {
      case TInt32 => TInt32
      case TInt64 => TInt64
      case TFloat32 => TFloat32
      case TFloat64 => TFloat64
      case _ => throw new RuntimeException(s"$op cannot be applied to $t")
    }
    case Bang() => t match {
      case TBoolean => TBoolean
    }
  }

  def compile(op: UnaryOp, t: Type, x: Code[_]): Code[_] = t match {
    case TBoolean =>
      val xx = coerce[Boolean](x)
      op match {
        case Bang() => !xx
      }
    case TInt32 =>
      val xx = coerce[Int](x)
      op match {
        case Negate() => -xx
      }
    case TInt64 =>
      val xx = coerce[Long](x)
      op match {
        case Negate() => -xx
      }
    case TFloat32 =>
      val xx = coerce[Float](x)
      op match {
        case Negate() => -xx
      }
    case TFloat64 =>
      val xx = coerce[Double](x)
      op match {
        case Negate() => -xx
      }
  }
}

sealed trait UnaryOp { }
case class Negate() extends UnaryOp { }
case class Bang() extends UnaryOp { }
