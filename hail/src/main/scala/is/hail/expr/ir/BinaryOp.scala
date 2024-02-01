package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.types.physical.{typeToTypeInfo, PType}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._

object BinaryOp {
  private val returnType: ((BinaryOp, Type, Type)) => Option[Type] = lift {
    case (FloatingPointDivide(), TInt32, TInt32) => TFloat64
    case (FloatingPointDivide(), TInt64, TInt64) => TFloat64
    case (FloatingPointDivide(), TFloat32, TFloat32) => TFloat32
    case (FloatingPointDivide(), TFloat64, TFloat64) => TFloat64
    case (
          Add() | Subtract() | Multiply() | RoundToNegInfDivide() | BitAnd() | BitOr() | BitXOr(),
          TInt32,
          TInt32,
        ) => TInt32
    case (
          Add() | Subtract() | Multiply() | RoundToNegInfDivide() | BitAnd() | BitOr() | BitXOr(),
          TInt64,
          TInt64,
        ) => TInt64
    case (Add() | Subtract() | Multiply() | RoundToNegInfDivide(), TFloat32, TFloat32) => TFloat32
    case (Add() | Subtract() | Multiply() | RoundToNegInfDivide(), TFloat64, TFloat64) => TFloat64
    case (LeftShift() | RightShift() | LogicalRightShift(), t @ (TInt32 | TInt64), TInt32) => t
  }

  def defaultDivideOp(t: Type): BinaryOp = t match {
    case TInt32 | TInt64 => RoundToNegInfDivide()
    case TFloat32 | TFloat64 => FloatingPointDivide()
  }

  def returnTypeOption(op: BinaryOp, l: Type, r: Type): Option[Type] =
    returnType((op, l, r))

  def getReturnType(op: BinaryOp, l: Type, r: Type): Type =
    returnType((op, l, r)).getOrElse(incompatible(l, r, op))

  private def incompatible[T](lt: Type, rt: Type, op: BinaryOp): T =
    throw new RuntimeException(s"Cannot apply $op to $lt and $rt")

  def emit(cb: EmitCodeBuilder, op: BinaryOp, l: SValue, r: SValue): SValue = {
    val lt = l.st.virtualType
    val rt = r.st.virtualType

    val retCode = emit(op, lt, rt, SType.extractPrimValue(cb, l), SType.extractPrimValue(cb, r))
    val retT = getReturnType(op, lt, rt)
    primitive(retT, cb.memoizeAny(retCode, typeToTypeInfo(PType.canonical(retT))))
  }

  private[this] def emit(op: BinaryOp, lt: Type, rt: Type, l: Code[_], r: Code[_]): Code[_] =
    (lt, rt) match {
      case (TInt32, TInt32) =>
        val ll = coerce[Int](l)
        val rr = coerce[Int](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll.toD / rr.toD
          case RoundToNegInfDivide() => Code.invokeStatic2[Math, Int, Int, Int]("floorDiv", ll, rr)
          case BitAnd() => ll & rr
          case BitOr() => ll | rr
          case BitXOr() => ll ^ rr
          case LeftShift() => ll << rr
          case RightShift() => ll >> rr
          case LogicalRightShift() => ll >>> rr
          case _ => incompatible(lt, rt, op)
        }
      case (TInt64, TInt32) =>
        val ll = coerce[Long](l)
        val rr = coerce[Int](r)
        op match {
          case LeftShift() => ll << rr
          case RightShift() => ll >> rr
          case LogicalRightShift() => ll >>> rr
          case _ => incompatible(lt, rt, op)
        }
      case (TInt64, TInt64) =>
        val ll = coerce[Long](l)
        val rr = coerce[Long](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll.toD / rr.toD
          case RoundToNegInfDivide() =>
            Code.invokeStatic2[Math, Long, Long, Long]("floorDiv", ll, rr)
          case BitAnd() => ll & rr
          case BitOr() => ll | rr
          case BitXOr() => ll ^ rr
          case _ => incompatible(lt, rt, op)
        }
      case (TFloat32, TFloat32) =>
        val ll = coerce[Float](l)
        val rr = coerce[Float](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll / rr
          case RoundToNegInfDivide() =>
            Code.invokeStatic1[Math, Double, Double]("floor", ll.toD / rr.toD).toF
          case _ => incompatible(lt, rt, op)
        }
      case (TFloat64, TFloat64) =>
        val ll = coerce[Double](l)
        val rr = coerce[Double](r)
        op match {
          case Add() => ll + rr
          case Subtract() => ll - rr
          case Multiply() => ll * rr
          case FloatingPointDivide() => ll / rr
          case RoundToNegInfDivide() => Code.invokeStatic1[Math, Double, Double]("floor", ll / rr)
          case _ => incompatible(lt, rt, op)
        }
      case (TBoolean, TBoolean) =>
        op match {
          case _ => incompatible(lt, rt, op)
        }

      case _ => incompatible(lt, rt, op)
    }

  val fromString: PartialFunction[String, BinaryOp] = {
    case "+" | "Add" => Add()
    case "-" | "Subtract" => Subtract()
    case "*" | "Multiply" => Multiply()
    case "/" | "FloatingPointDivide" => FloatingPointDivide()
    case "//" | "RoundToNegInfDivide" => RoundToNegInfDivide()
    case "|" | "BitOr" => BitOr()
    case "&" | "BitAnd" => BitAnd()
    case "^" | "BitXOr" => BitXOr()
    case "<<" | "LeftShift" => LeftShift()
    case ">>" | "RightShift" => RightShift()
    case ">>>" | "LogicalRightShift" => LogicalRightShift()
  }
}

sealed trait BinaryOp {}

case class Add() extends BinaryOp {}

case class Subtract() extends BinaryOp {}

case class Multiply() extends BinaryOp {}

case class FloatingPointDivide() extends BinaryOp {}

case class RoundToNegInfDivide() extends BinaryOp {}

case class BitAnd() extends BinaryOp

case class BitOr() extends BinaryOp

case class BitXOr() extends BinaryOp

case class LeftShift() extends BinaryOp

case class RightShift() extends BinaryOp

case class LogicalRightShift() extends BinaryOp
