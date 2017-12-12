package is.hail.expr.ir

import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr._

object BinaryOp {
  def inferReturnType(op: BinaryOp, l: Type, r: Type): Type = op match {
    case Add() | Subtract() | Multiply() | Divide() => (l, r) match {
      case (_: TInt32, _: TInt32) => TInt32()
      case (_: TInt64, _: TInt64) => TInt64()
      case (_: TFloat32, _: TFloat32) => TFloat32()
      case (_: TFloat64, _: TFloat64) => TFloat64()
      case _ => incompatible(l, r, op)
    }
    case GT() | GTEQ() | LTEQ() | LT() => (l, r) match {
      case (_: TInt32, _: TInt32) => TBoolean()
      case (_: TInt64, _: TInt64) => TBoolean()
      case (_: TFloat32, _: TFloat32) => TBoolean()
      case (_: TFloat64, _: TFloat64) => TBoolean()
      case _ => incompatible(l, r, op)
    }
    case EQ() | NEQ() => (l, r) match {
      case (_: TInt32, _: TInt32) => TBoolean()
      case (_: TInt64, _: TInt64) => TBoolean()
      case (_: TFloat32, _: TFloat32) => TBoolean()
      case (_: TFloat64, _: TFloat64) => TBoolean()
      case (_: TBoolean, _: TBoolean) => TBoolean()
      case _ => incompatible(l, r, op)
    }
    case DoubleAmpersand() | DoublePipe() => (l, r) match {
      case (_: TBoolean, _: TBoolean) => TBoolean()
      case _ => incompatible(l, r, op)
    }
  }

  private def incompatible[T](lt: Type, rt: Type, op: BinaryOp): T =
    fatal(s"Cannot apply $op to $lt and $rt")

  def compile(op: BinaryOp, lt: Type, rt: Type, l: Code[_], r: Code[_]): Code[_] = (lt, rt) match {
    case (_: TInt32, _: TInt32) =>
      val ll = coerce[Int](l)
      val rr = coerce[Int](r)
      op match {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case Divide() => ll / rr
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
        case _ => incompatible(lt, rt, op)
      }
    case (_: TInt64, _: TInt64) =>
      val ll = coerce[Long](l)
      val rr = coerce[Long](r)
      op match {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case Divide() => ll / rr
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
        case _ => incompatible(lt, rt, op)
      }
    case (_: TFloat32, _: TFloat32) =>
      val ll = coerce[Float](l)
      val rr = coerce[Float](r)
      op match {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case Divide() => ll / rr
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
        case _ => incompatible(lt, rt, op)
      }
    case (_: TFloat64, _: TFloat64) =>
      val ll = coerce[Double](l)
      val rr = coerce[Double](r)
      op match {
        case Add() => ll + rr
        case Subtract() => ll - rr
        case Multiply() => ll * rr
        case Divide() => ll / rr
        case GT() => ll > rr
        case GTEQ() => ll >= rr
        case LTEQ() => ll <= rr
        case LT() => ll < rr
        case EQ() => ll.ceq(rr)
        case NEQ() => ll.cne(rr)
        case _ => incompatible(lt, rt, op)
      }
    case (_: TBoolean, _: TBoolean) =>
      val ll = coerce[Boolean](l)
      val rr = coerce[Boolean](r)
      op match {
        case DoubleAmpersand() => ll && rr
        case DoublePipe() => ll || rr
        case EQ() => ll.toI.ceq(rr.toI)
        case NEQ() => ll.toI ^ rr.toI
        case _ => incompatible(lt, rt, op)
      }

    case _ => incompatible(lt, rt, op)
  }

  val fromString: PartialFunction[String, BinaryOp] = {
    case "+" => Add()
    case "-" => Subtract()
    case "*" => Multiply()
    case "/" => Divide()
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

sealed trait BinaryOp { }
case class Add() extends BinaryOp { }
case class Subtract() extends BinaryOp { }
case class Multiply() extends BinaryOp { }
case class Divide() extends BinaryOp { }
case class GT() extends BinaryOp { }
case class GTEQ() extends BinaryOp { }
case class LTEQ() extends BinaryOp { }
case class LT() extends BinaryOp { }
case class EQ() extends BinaryOp { }
case class NEQ() extends BinaryOp { }
case class DoubleAmpersand() extends BinaryOp { }
case class DoublePipe() extends BinaryOp { }

