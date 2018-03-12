package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._

object UtilFunctions extends RegistryFunctions {
  registerCode[Int]("triangle", TInt32(), TInt32()) { case (_, Array(n: Code[Int])) => n * (n + 1) / 2 }

  registerIR("size", TArray(TInt32()), TInt32()) { case Seq(a) => ArrayLen(a) }

  registerIR("size", TArray(TInt64()), TInt32()) { case Seq(a) => ArrayLen(a) }

  for (zero <- Seq(I32(0), I64(0), F32(0), F64(0))) {
    registerIR("sum", TArray(zero.typ), zero.typ) {
      case Seq(a) =>
        ArrayFold(a, zero, "sum", "v", ApplyBinaryPrimOp(Add(), Ref("sum"), If(IsNA(Ref("v")), zero, Ref("v"))), typ = zero.typ)
    }
  }

  for (t <- Seq(TInt32(), TInt64(), TFloat32(), TFloat64())) {
    registerIR("min", TArray(t), t) {
      case Seq(a) =>
        val na = NA(t)
        val min = Ref("min")
        val value = Ref("v")
        val body = If(IsNA(min), value, If(IsNA(value), min, If(ApplyBinaryPrimOp(LT(), min, value), min, value)))
        ArrayFold(a, na, "min", "v", body)
    }
  }
}
