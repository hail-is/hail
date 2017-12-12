package is.hail.expr.ir

import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr._

object UnaryOp {

  private val returnType: ((UnaryOp, Type)) => Option[Type] = lift {
    case (Negate(), t@(_: TInt32 | _: TInt64 | _: TFloat32 | _: TFloat64)) => t
    case (Bang(), t: TBoolean) => t
  }

  def returnTypeOption(op: UnaryOp, t: Type): Option[Type] =
    returnType(op, t)

  def getReturnType(op: UnaryOp, t: Type): Type =
    returnType(op, t).getOrElse(incompatible(t, op))

  private def incompatible[T](t: Type, op: UnaryOp): T =
    throw new RuntimeException(s"Cannot apply $op to values of type $t")

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
