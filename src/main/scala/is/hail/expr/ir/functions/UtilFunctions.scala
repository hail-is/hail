package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.coerce

object UtilFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("triangle", TInt32(), TInt32()) { (_, n: Code[Int]) => n * (n + 1) / 2 }

    registerIR("size", TArray(tv("T")))(ArrayLen)

    registerIR("sum", TArray(tnum("T"))) { a =>
      val zero = Literal(0, coerce[TArray](a.typ).elementType)
      ArrayFold(a, zero, "sum", "v", If(IsNA(Ref("v")), Ref("sum"), ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))))
    }

    registerIR("sum", TAggregable(tv("T")))(ApplyAggOp(_, Sum(), Seq()))

    registerIR("min", TArray(tnum("T"))) { a =>
      val body = If(IsNA(Ref("min")),
        Ref("value"),
        If(IsNA(Ref("value")),
          Ref("min"),
          If(ApplyBinaryPrimOp(LT(), Ref("min"), Ref("value")), Ref("min"), Ref("value"))))
      ArrayFold(a, NA(tnum("T").t), "min", "value", body)
    }

    registerIR("isDefined", tv("T")) { a => ApplyUnaryPrimOp(Bang(), IsNA(a)) }

    registerIR("orMissing", TBoolean(), tv("T")) { (cond, a) => If(cond, a, NA(tv("T").t)) }

    registerIR("[]", TArray(tv("T")), TInt32()) { (a, i) => ArrayRef(a, i) }

    registerIR("[]", tv("T", _.isInstanceOf[TTuple]), TInt32()) { (a, i) => GetTupleElement(a, i.asInstanceOf[I32].x) }

    registerIR("range", TInt32(), TInt32(), TInt32())(ArrayRange)

    registerIR("range", TInt32(), TInt32())(ArrayRange(_, _, I32(1)))

    registerIR("range", TInt32())(ArrayRange(I32(0), _, I32(1)))
  }
}
