package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.expr.types.coerce

object UtilFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("triangle", TInt32(), TInt32()) { (_, n: Code[Int]) => n * (n + 1) / 2 }

    registerIR("size", TArray(tv("T")))(ArrayLen)

    registerIR("sum", TArray(tnum("T"))) { a =>
      val zero = Literal(0, coerce[TArray](a.typ).elementType)
      ArrayFold(a, zero, "sum", "v", If(IsNA(Ref("v")), Ref("sum"), ApplyBinaryPrimOp(Add(), Ref("sum"), Ref("v"))))
    }


    registerIR("sum", TAggregable(tnum("T")))(ApplyAggOp(_, Sum(), FastSeq()))

    registerIR("product", TArray(tnum("T"))) { a =>
      val product = genUID()
      val v = genUID()
      val one = Literal(1, coerce[TArray](a.typ).elementType)
      ArrayFold(a, one, product, v, If(IsNA(Ref(v)), Ref(product), ApplyBinaryPrimOp(Multiply(), Ref(product), Ref(v))))
    }

    registerIR("count", TAggregable(tv("T"))) { agg =>
      val uid = genUID()
      ApplyAggOp(AggMap(agg, uid, I32(0)), Count(), Seq())
    }

    registerIR("min", TArray(tnum("T"))) { a =>
      val min = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(min)),
        Ref(value),
        If(IsNA(Ref(value)),
          Ref(min),
          If(ApplyBinaryPrimOp(LT(), Ref(value), Ref(min)), Ref(value), Ref(min))))
      ArrayFold(a, NA(tnum("T").t), min, value, body)
    }

    registerIR("max", TArray(tnum("T"))) { a =>
      val max = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(max)),
        Ref(value),
        If(IsNA(Ref(value)),
          Ref(max),
          If(ApplyBinaryPrimOp(GT(), Ref(value), Ref(max)), Ref(value), Ref(max))))
      ArrayFold(a, NA(tnum("T").t), max, value, body)
    }

    registerIR("isDefined", tv("T")) { a => ApplyUnaryPrimOp(Bang(), IsNA(a)) }
    registerIR("isMissing", tv("T")) { a => IsNA(a) }

    registerIR("[]", TArray(tv("T")), TInt32()) { (a, i) => ArrayRef(a, i) }

    registerIR("[:]", TArray(tv("T"))) { (a) => a }

    registerIR("[*:]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(If(ApplyBinaryPrimOp(LT(), i, I32(0)),
          ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
          i),
          ArrayLen(a),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx)))
    }

    registerIR("[:*]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(I32(0),
          If(ApplyBinaryPrimOp(LT(), i, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
            i),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx)))
    }

    registerIR("[*:*]", TArray(tv("T")), TInt32(), TInt32()) { (a, i, j) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(
          If(ApplyBinaryPrimOp(LT(), i, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
            i),
          If(ApplyBinaryPrimOp(LT(), j, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), j),
            j),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx)))
    }

    registerIR("[]", tv("T", _.isInstanceOf[TTuple]), TInt32()) { (a, i) => GetTupleElement(a, i.asInstanceOf[I32].x) }

    registerIR("[:]", TString()) { (a) => a }

    registerIR("range", TInt32(), TInt32(), TInt32())(ArrayRange)

    registerIR("range", TInt32(), TInt32())(ArrayRange(_, _, I32(1)))

    registerIR("range", TInt32())(ArrayRange(I32(0), _, I32(1)))

    registerIR("annotate", tv("T", _.isInstanceOf[TStruct]), tv("U", _.isInstanceOf[TStruct])) { (s, annotations) =>
      InsertFields(s, annotations.asInstanceOf[MakeStruct].fields)
    }
  }
}
