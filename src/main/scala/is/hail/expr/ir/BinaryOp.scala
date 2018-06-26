package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

object BinaryOp {
  private val returnType: ((BinaryOp, Type, Type)) => Option[Type] = lift {
    case (FloatingPointDivide(), _: TInt32, _: TInt32) => TFloat32()
    case (FloatingPointDivide(), _: TInt64, _: TInt64) => TFloat32()
    case (FloatingPointDivide(), _: TFloat32, _: TFloat32) => TFloat32()
    case (FloatingPointDivide(), _: TFloat64, _: TFloat64) => TFloat64()
    case (Add() | Subtract() | Multiply() | RoundToNegInfDivide(), _: TInt32, _: TInt32) => TInt32()
    case (Add() | Subtract() | Multiply() | RoundToNegInfDivide(), _: TInt64, _: TInt64) => TInt64()
    case (Add() | Subtract() | Multiply() | RoundToNegInfDivide(), _: TFloat32, _: TFloat32) => TFloat32()
    case (Add() | Subtract() | Multiply() | RoundToNegInfDivide(), _: TFloat64, _: TFloat64) => TFloat64()
  }

  def defaultDivideOp(t: Type): BinaryOp = t match {
    case _: TInt32 | _: TInt64 => RoundToNegInfDivide()
    case _: TFloat32 | _: TFloat64 => FloatingPointDivide()
  }

  def returnTypeOption(op: BinaryOp, l: Type, r: Type): Option[Type] =
    returnType(op, l, r)

  def getReturnType(op: BinaryOp, l: Type, r: Type): Type =
    returnType(op, l, r).getOrElse(incompatible(l, r, op))

  private def incompatible[T](lt: Type, rt: Type, op: BinaryOp): T =
    throw new RuntimeException(s"Cannot apply $op to $lt and $rt")

  def emit(op: BinaryOp, lt: Type, rt: Type, l: Code[_], r: Code[_]): Code[_] =
    (lt, rt) match {
      case (_: TInt32, _: TInt32) =>
        val ll = coerce[Int](l)
        val rr = coerce[Int](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll.toF / rr.toF
          case RoundToNegInfDivide() => Code.invokeStatic[Math, Int, Int, Int]("floorDiv", ll, rr)
          case _ => incompatible(lt, rt, op)
        }
      case (_: TInt64, _: TInt64) =>
        val ll = coerce[Long](l)
        val rr = coerce[Long](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll.toF / rr.toF
          case RoundToNegInfDivide() => Code.invokeStatic[Math, Long, Long, Long]("floorDiv", ll, rr)
          case _ => incompatible(lt, rt, op)
        }
      case (_: TFloat32, _: TFloat32) =>
        val ll = coerce[Float](l)
        val rr = coerce[Float](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll / rr
          case RoundToNegInfDivide() => Code.invokeStatic[Math, Double, Double]("floor", ll.toD / rr.toD).toF
          case _ => incompatible(lt, rt, op)
        }
      case (_: TFloat64, _: TFloat64) =>
        val ll = coerce[Double](l)
        val rr = coerce[Double](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll / rr
          case RoundToNegInfDivide() => Code.invokeStatic[Math, Double, Double]("floor", ll / rr)
          case _ => incompatible(lt, rt, op)
        }
      case (_: TBoolean, _: TBoolean) =>
        val ll = coerce[Boolean](l)
        val rr = coerce[Boolean](r)
        op match {
          case _ => incompatible(lt, rt, op)
        }

      case _ => incompatible(lt, rt, op)
    }

  val fromString: PartialFunction[String, BinaryOp] = {
    case "+" | "Add" => Add()
    case "-" | "Subtract" => Subtract()
    case "*" | "Multiply" => Multiply()
    case "/" | "FloatingPointDivide"  => FloatingPointDivide()
    case "//" | "RoundToNegInfDivide" => RoundToNegInfDivide()
  }
}

sealed trait BinaryOp {}

case class Add() extends BinaryOp {}

case class Subtract() extends BinaryOp {}

case class Multiply() extends BinaryOp {}

case class FloatingPointDivide() extends BinaryOp {}

case class RoundToNegInfDivide() extends BinaryOp {}
