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
    case (GT() | GTEQ() | LTEQ() | LT(), _: TInt32, _: TInt32) => TBoolean()
    case (GT() | GTEQ() | LTEQ() | LT(), _: TInt64, _: TInt64) => TBoolean()
    case (GT() | GTEQ() | LTEQ() | LT(), _: TFloat32, _: TFloat32) => TBoolean()
    case (GT() | GTEQ() | LTEQ() | LT(), _: TFloat64, _: TFloat64) => TBoolean()
    case (EQ() | NEQ(), _: TInt32, _: TInt32) => TBoolean()
    case (EQ() | NEQ(), _: TInt64, _: TInt64) => TBoolean()
    case (EQ() | NEQ(), _: TFloat32, _: TFloat32) => TBoolean()
    case (EQ() | NEQ(), _: TFloat64, _: TFloat64) => TBoolean()
    case (EQ() | NEQ(), _: TBoolean, _: TBoolean) => TBoolean()
    case (DoubleAmpersand() | DoublePipe(), _: TBoolean, _: TBoolean) => TBoolean()
  }

  def returnTypeOption(op: BinaryOp, l: Type, r: Type): Option[Type] =
    returnType(op, l, r)

  def getReturnType(op: BinaryOp, l: Type, r: Type): Type =
    returnType(op, l, r).getOrElse(incompatible(l, r, op))

  private def incompatible[T](lt: Type, rt: Type, op: BinaryOp): T =
    throw new RuntimeException(s"Cannot apply $op to $lt and $rt")

  def emit(fb: FunctionBuilder[_], mb: StagedBitSet, op: BinaryOp, lt: Type, rt: Type, lm: Code[Boolean], lv: Code[_], rm: Code[Boolean], rv: Code[_]): (Code[Unit], Code[Boolean], Code[_]) = {
    emitPropagatesNA(op, lt, rt, lv, rv).map {
      (Code._empty, lm || rm, _)
    }
      .orElse({
        val xlm = mb.newBit()
        val xrm = mb.newBit()
        (lt, rt) match {
          case (_: TBoolean, _: TBoolean) =>
            val ll = coerce[Boolean](lv)
            val rr = coerce[Boolean](rv)
            val xlv = mb.newBit()
            val xrv = mb.newBit()
            optMatch(op) {
              case DoubleAmpersand() =>
                (Code(xlm := lm, xrm := rm, xlv := coerce[Boolean](xlm.mux(defaultValue(TBoolean()), ll))), xlm || (xlv && xrm), xlv && rr)
              case DoublePipe() =>
                (Code(xlm := lm, xrm := rm, xlv := coerce[Boolean](xlm.mux(defaultValue(TBoolean()), ll))), xlm || (!xlv && xrm), xlv || rr)
            }
          case _ => None
        }
      }).getOrElse(incompatible(lt, rt, op))
  }

  def emitPropagatesNA(op: BinaryOp, lt: Type, rt: Type, l: Code[_], r: Code[_]): Option[Code[_]] = (lt, rt) match {
    case (_: TInt32, _: TInt32) =>

      val ll = coerce[Int](l)
      val rr = coerce[Int](r)
      optMatch(op) {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case FloatingPointDivide() => ll.toF / rr.toF
        case RoundToNegInfDivide() => Code.invokeStatic[Math, Int, Int, Int]("floorDiv", ll, rr)
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
      }
    case (_: TInt64, _: TInt64) =>
      val ll = coerce[Long](l)
      val rr = coerce[Long](r)
      optMatch(op) {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case FloatingPointDivide() => ll.toF / rr.toF
        case RoundToNegInfDivide() => Code.invokeStatic[Math, Long, Long, Long]("floorDiv", ll, rr)
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
      }
    case (_: TFloat32, _: TFloat32) =>
      val ll = coerce[Float](l)
      val rr = coerce[Float](r)
      optMatch(op) {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case FloatingPointDivide() => ll / rr
        case RoundToNegInfDivide() => Code.invokeStatic[Math, Float, Float]("floor", ll / rr)
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
      }
    case (_: TFloat64, _: TFloat64) =>
      val ll = coerce[Double](l)
      val rr = coerce[Double](r)
      optMatch(op) {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case FloatingPointDivide() => ll / rr
        case RoundToNegInfDivide() => Code.invokeStatic[Math, Double, Double]("floor", ll / rr)
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
      }
    case (_: TBoolean, _: TBoolean) =>
      val ll = coerce[Boolean](l)
      val rr = coerce[Boolean](r)
      optMatch(op) {
        case EQ() => ll.toI.ceq(rr.toI)
        case NEQ() => ll.toI ^ rr.toI
      }

    case _ => None
  }

  val fromString: PartialFunction[String, BinaryOp] = {
    case "+" => Add()
    case "-" => Subtract()
    case "*" => Multiply()
    case "/" => FloatingPointDivide()
    case "//" => RoundToNegInfDivide()
    case ">" => GT()
    case ">=" => GTEQ()
    case "<=" => LTEQ()
    case "<" => LT()
    case "==" => EQ()
    case "!=" => NEQ()
    case "&&" => DoubleAmpersand()
    case "||" => DoublePipe()
  }
}

sealed trait BinaryOp {}

case class Add() extends BinaryOp {}
case class Subtract() extends BinaryOp {}
case class Multiply() extends BinaryOp {}
case class FloatingPointDivide() extends BinaryOp {}
case class RoundToNegInfDivide() extends BinaryOp {}
case class GT() extends BinaryOp {}
case class GTEQ() extends BinaryOp {}
case class LTEQ() extends BinaryOp {}
case class LT() extends BinaryOp {}
case class EQ() extends BinaryOp {}
case class NEQ() extends BinaryOp {}
case class DoubleAmpersand() extends BinaryOp {}
case class DoublePipe() extends BinaryOp {}
