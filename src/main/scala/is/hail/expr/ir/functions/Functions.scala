package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.{AST, Lambda}
import is.hail.expr.ir.{IR, _}
import is.hail.expr.types._
import is.hail.expr.types.coerce
import is.hail.utils._

import scala.collection.mutable

object IRFunctionRegistry {
  val registry: mutable.Map[String, IndexedSeq[IRFunction[_]]] = mutable.Map()

  def addFunction(name: String, f: IRFunction[_]) {
    val updated = registry.getOrElse(name, IndexedSeq()) :+ f
    registry.put(name, updated)
  }

  def lookupFunction(name: String, args: Seq[Type]): Option[IRFunction[_]] = {
    registry.get(name).flatMap { funcs =>
//      warn(s"$name found w/ args {${funcs.map{f => f.types.mkString(",")}.mkString("}; {")}}; expected: ${args.mkString(",")}")
      val method = funcs.filter { f => f.types.zip(args).forall { case (t1, t2) => t1.isOfType(t2) } }
      method match {
        case IndexedSeq() => None
        case IndexedSeq(x) => Some(x)
        case IndexedSeq(x, y) => if (x.types.zip(args).forall {case (t1, t2) => t1 == t2 }) Some(x) else Some(y)
      }
    }
  }
}

object IRRegistry {
  val registry: mutable.Map[String, IR] = mutable.Map()

  val functions = Array(
    new CallFunctions(),
    new UtilFunctions(),
    new GenotypeFunctions())

  private def tryPrimOpConversion(fn: String): IndexedSeq[IR] => Option[IR] =
    flatLift {
      case IndexedSeq(x) => for {
        op <- UnaryOp.fromString.lift(fn)
        t <- UnaryOp.returnTypeOption(op, x.typ)
      } yield ApplyUnaryPrimOp(op, x, t)
      case IndexedSeq(x, y) => for {
        op <- BinaryOp.fromString.lift(fn)
        t <- BinaryOp.returnTypeOption(op, x.typ, y.typ)
      } yield ApplyBinaryPrimOp(op, x, y, t)
    }

  def lookup(name: String, args: IndexedSeq[IR], types: IndexedSeq[Type]): Option[IR] = {
    (name, args) match {
      case ("range", IndexedSeq(a, b, c)) => Some(ArrayRange(a, b, c))
      case ("isDefined", IndexedSeq(x)) => Some(ApplyUnaryPrimOp(Bang(), IsNA(x)))
      case ("orMissing", IndexedSeq(cond, x)) => Some(If(cond, x, NA(x.typ)))
      case ("size", IndexedSeq(x)) if x.typ.isInstanceOf[TArray] => Some(ArrayLen(x))
      case ("sum", IndexedSeq(x)) if x.typ.isInstanceOf[TArray] =>
          val zero = coerce[TArray](x.typ).elementType match {
            case _: TInt32 => I32(0)
            case _: TInt64 => I64(0)
            case _: TFloat32 => F32(0)
            case _: TFloat64 => F64(0)
          }
        Some(ArrayFold(x, zero, "sum", "v", ApplyBinaryPrimOp(Add(), Ref("sum"), If(IsNA(Ref("v")), zero, Ref("v"))), typ=zero.typ))
      case ("min", IndexedSeq(x)) if x.typ.isInstanceOf[TArray] =>
        val na = NA(coerce[TArray](x.typ).elementType)
        val min = Ref("min")
        val value = Ref("v")
        val body = If(IsNA(min), value, If(IsNA(value), min, If(ApplyBinaryPrimOp(LT(), min, value), min, value)))
        Some(ArrayFold(x, na, "min", "v", body))
//
//      case ("gqFromPL", IndexedSeq(x)) =>
//        Some(GenotypeFunctions.gqFromPL(x))

      case (n, a) =>
        tryPrimOpConversion(name)(a).orElse(
          IRFunctionRegistry.lookupFunction(n, types)
            .map { irf => ApplyFunction(irf, a) })
    }
  }

  def convertLambda(method: String, lhs: AST, lambda: Lambda, agg: Option[String]): Option[IR] = {
    (lhs.`type`, method) match {
      case (t: TArray, "map") =>
        for {
          ir <- lhs.toIR(agg)
          body <- lambda.body.toIR(agg)
        } yield {
          ArrayMap(ir, lambda.param, body, t.elementType)
        }
      case (t: TArray, "filter") =>
        for {
          ir <- lhs.toIR(agg)
          body <- lambda.body.toIR(agg)
        } yield {
          ArrayFilter(ir, lambda.param, body)
        }
    }
  }
}

object IRFunction {
  def apply[T](mname: String, t: Type*)(impl: (MethodBuilder, Array[Code[_]]) => Code[T]): IRFunction[T] = {
    val f = new IRFunction[T] {
      override val name: String = mname

      override val types: Array[Type] = t.toArray

      override val implementation: (MethodBuilder, Array[Code[_]]) => Code[T] = impl
    }

    IRFunctionRegistry.addFunction(mname, f)
    f
  }
}

abstract class IRFunction[T] {
  def name: String

  def types: Array[Type]

  def implementation: (MethodBuilder, Array[Code[_]]) => Code[T]

  def apply(mb: MethodBuilder, args: Code[_]*): Code[T] = implementation(mb, args.toArray)

  def returnType: Type = types.last
}