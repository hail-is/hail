package is.hail.expr.ir

import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr._

object UnaryOp {
  def inferReturnType(op: UnaryOp, t: Type): Type = op match {
    case Negate() => t match {
      case _: TInt32 => TInt32()
      case _: TInt64 => TInt64()
      case _: TFloat32 => TFloat32()
      case _: TFloat64 => TFloat64()
      case _ => incompatible(t, op)
    }
    case Bang() => t match {
      case _: TBoolean => TBoolean()
      case _ => incompatible(t, op)
    }
  }

  private def incompatible[T](t: Type, op: UnaryOp): T =
    fatal(s"Cannot apply $op to values of type $t")

  def compile(op: UnaryOp, t: Type, x: Code[_]): Code[_] = t match {
    case _: TBoolean =>
      val xx = coerce[Boolean](x)
      op match {
        case Bang() => !xx
        case _ => incompatible(t, op)
      }
    case _: TInt32 =>
      val xx = coerce[Int](x)
      op match {
        case Negate() => -xx
        case _ => incompatible(t, op)
      }
    case _: TInt64 =>
      val xx = coerce[Long](x)
      op match {
        case Negate() => -xx
        case _ => incompatible(t, op)
      }
    case _: TFloat32 =>
      val xx = coerce[Float](x)
      op match {
        case Negate() => -xx
        case _ => incompatible(t, op)
      }
    case _: TFloat64 =>
      val xx = coerce[Double](x)
      op match {
        case Negate() => -xx
        case _ => incompatible(t, op)
      }
    case _ => incompatible(t, op)
  }

  val fromString: PartialFunction[String, UnaryOp] = {
    case "-" => Negate()
    case "!" => Bang()
  }
}

sealed trait UnaryOp { }
case class Negate() extends UnaryOp { }
case class Bang() extends UnaryOp { }
