package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.coerce

object UtilFunctions extends RegistryFunctions {
  registerCode[Int]("triangle", TInt32(), TInt32()) { case (_, n: Code[Int]) => n * (n + 1) / 2 }

  registerIR("size", TArray(tv("T")), TInt32())(ArrayLen)

  registerIR("sum", TArray(tv("T")), tv("T")) { a =>
    val zero = coerce[TArray](a.typ).elementType match {
      case _: TInt32 => I32(0)
      case _: TInt64 => I64(0)
      case _: TFloat32 => F32(0)
      case _: TFloat64 => F64(0)
    }
    ArrayFold(a, zero, "sum", "v", ApplyBinaryPrimOp(Add(), Ref("sum"), If(IsNA(Ref("v")), zero, Ref("v"))), typ = zero.typ)
  }

  registerIR("min", TArray(tv("T")), tv("T")) { a =>
    val na = NA(coerce[TArray](a.typ).elementType)
    val min = Ref("min")
    val value = Ref("v")
    val body = If(IsNA(min), value, If(IsNA(value), min, If(ApplyBinaryPrimOp(LT(), min, value), min, value)))
    ArrayFold(a, na, "min", "v", body)
  }
}
