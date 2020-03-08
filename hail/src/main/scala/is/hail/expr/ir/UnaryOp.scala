package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.utils._

object UnaryOp {

  private val returnType: ((UnaryOp, Type)) => Option[Type] = lift {
    case (Negate(), t@(TInt32 | TInt64 | TFloat32 | TFloat64)) => t
    case (Bang(), TBoolean) => TBoolean
    case (BitNot(), t@(TInt32 | TInt64)) => t
  }

  def returnTypeOption(op: UnaryOp, t: Type): Option[Type] =
    returnType((op, t))

  def getReturnType(op: UnaryOp, t: Type): Type =
    returnType((op, t)).getOrElse(incompatible(t, op))

  private def incompatible[T](t: Type, op: UnaryOp): T =
    throw new RuntimeException(s"Cannot apply $op to values of type $t")

  def emit(op: UnaryOp, t: Type, x: Code[_]): Code[_] = t match {
    case TBoolean =>
      val xx = coerce[Boolean](x)
      op match {
        case Bang() => !xx
        case _ => incompatible(t, op)
      }
    case TInt32 =>
      val xx = coerce[Int](x)
      op match {
        case Negate() => -xx
        case BitNot() => ~xx
        case _ => incompatible(t, op)
      }
    case TInt64 =>
      val xx = coerce[Long](x)
      op match {
        case Negate() => -xx
        case BitNot() => ~xx
        case _ => incompatible(t, op)
      }
    case TFloat32 =>
      val xx = coerce[Float](x)
      op match {
        case Negate() => -xx
        case _ => incompatible(t, op)
      }
    case TFloat64 =>
      val xx = coerce[Double](x)
      op match {
        case Negate() => -xx
        case _ => incompatible(t, op)
      }
    case _ => incompatible(t, op)
  }

  val fromString: PartialFunction[String, UnaryOp] = {
    case "-" | "Negate" => Negate()
    case "!" | "Bang" => Bang()
    case "~" | "BitNot" => BitNot()
  }
}

sealed trait UnaryOp { }
case class Negate() extends UnaryOp { }
case class Bang() extends UnaryOp { }
case class BitNot() extends UnaryOp
