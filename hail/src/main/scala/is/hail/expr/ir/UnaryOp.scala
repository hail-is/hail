package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.types.physical.{typeToTypeInfo, PType}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._

object UnaryOp {

  private val returnType: ((UnaryOp, Type)) => Option[Type] = lift {
    case (Negate, t @ (TInt32 | TInt64 | TFloat32 | TFloat64)) => t
    case (Bang, TBoolean) => TBoolean
    case (BitNot, t @ (TInt32 | TInt64)) => t
    case (BitCount, TInt32 | TInt64) => TInt32
  }

  def returnTypeOption(op: UnaryOp, t: Type): Option[Type] =
    returnType((op, t))

  def getReturnType(op: UnaryOp, t: Type): Type =
    returnType((op, t)).getOrElse(incompatible(t, op))

  private def incompatible[T](t: Type, op: UnaryOp): T =
    throw new RuntimeException(s"Cannot apply $op to values of type $t")

  def emit(cb: EmitCodeBuilder, op: UnaryOp, x: SValue): SValue = {
    val retCode = emit(op, x.st.virtualType, SType.extractPrimValue(cb, x))
    val retT = getReturnType(op, x.st.virtualType)
    primitive(retT, cb.memoizeAny(retCode, typeToTypeInfo(PType.canonical(retT))))
  }

  private def emit(op: UnaryOp, t: Type, x: Code[_]): Code[_] = t match {
    case TBoolean =>
      val xx = coerce[Boolean](x)
      op match {
        case Bang => !xx
        case _ => incompatible(t, op)
      }
    case TInt32 =>
      val xx = coerce[Int](x)
      op match {
        case Negate => -xx
        case BitNot => ~xx
        case BitCount => xx.bitCount
        case _ => incompatible(t, op)
      }
    case TInt64 =>
      val xx = coerce[Long](x)
      op match {
        case Negate => -xx
        case BitNot => ~xx
        case BitCount => xx.bitCount
        case _ => incompatible(t, op)
      }
    case TFloat32 =>
      val xx = coerce[Float](x)
      op match {
        case Negate => -xx
        case _ => incompatible(t, op)
      }
    case TFloat64 =>
      val xx = coerce[Double](x)
      op match {
        case Negate => -xx
        case _ => incompatible(t, op)
      }
    case _ => incompatible(t, op)
  }

  val fromString: PartialFunction[String, UnaryOp] = {
    case "-" | "Negate" => Negate
    case "!" | "Bang" => Bang
    case "~" | "BitNot" => BitNot
    case "BitCount" => BitCount
  }
}

sealed trait UnaryOp
case object Negate extends UnaryOp
case object Bang extends UnaryOp
case object BitNot extends UnaryOp
case object BitCount extends UnaryOp
