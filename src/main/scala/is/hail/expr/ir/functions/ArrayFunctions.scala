package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.coerce

object ArrayFunctions extends RegistryFunctions {

  def registerAll() {
    registerIR("size", TArray(tv("T")))(ArrayLen)

    val arrayOps: Array[(String, (IR, IR) => IR)] =
      Array(
        ("*", ApplyBinaryPrimOp(Multiply(), _, _)),
        ("/", ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
        ("//", ApplyBinaryPrimOp(RoundToNegInfDivide(), _, _)),
        ("+", ApplyBinaryPrimOp(Add(), _, _)),
        ("-", ApplyBinaryPrimOp(Subtract(), _, _)),
        ("**", { (ir1: IR, ir2: IR) => Apply("**", Seq(ir1, ir2)) }),
        ("%", { (ir1: IR, ir2: IR) => Apply("%", Seq(ir1, ir2)) }))

    for ((stringOp, irOp) <- arrayOps) {
      registerIR(stringOp, TArray(tnum("T")), tv("T")) { (a, c) =>
        val i = genUID()
        ArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR(stringOp, tnum("T"), TArray(tv("T"))) { (c, a) =>
        val i = genUID()
        ArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR(stringOp, TArray(tnum("T")), TArray(tv("T"))) { (array1, array2) =>
        val a1id = genUID()
        val a1 = Ref(a1id, array1.typ)
        val a2id = genUID()
        val a2 = Ref(a2id, array2.typ)
        val iid = genUID()
        val i = Ref(iid, TInt32())
        val body =
          ArrayMap(ArrayRange(I32(0), ArrayLen(a1), I32(1)), iid,
            irOp(ArrayRef(a1, i), ArrayRef(a2, i)))
        Let(a1id, array1, Let(a2id, array2, body))
      }
    }

    registerIR("sum", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val sum = genUID()
      val v = genUID()
      val zero = Cast(I64(0), t)
      ArrayFold(a, zero, sum, v, If(IsNA(Ref(v, t)), Ref(sum, t), ApplyBinaryPrimOp(Add(), Ref(sum, t), Ref(v, t))))
    }

    registerIR("product", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val product = genUID()
      val v = genUID()
      val one = Cast(I64(1), t)
      ArrayFold(a, one, product, v, If(IsNA(Ref(v, t)), Ref(product, t), ApplyBinaryPrimOp(Multiply(), Ref(product, t), Ref(v, t))))
    }

    registerIR("min", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val min = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(min, t)),
        Ref(value, t),
        If(IsNA(Ref(value, t)),
          Ref(min, t),
          If(ApplyComparisonOp(LT(t), Ref(value, t), Ref(min, t)), Ref(value, t), Ref(min, t))))
      ArrayFold(a, NA(t), min, value, body)
    }

    registerIR("max", TArray(tnum("T"))) { a =>
      val t = -coerce[TArray](a.typ).elementType
      val max = genUID()
      val value = genUID()
      val body = If(IsNA(Ref(max, t)),
        Ref(value, t),
        If(IsNA(Ref(value, t)),
          Ref(max, t),
          If(ApplyComparisonOp(GT(t), Ref(value, t), Ref(max, t)), Ref(value, t), Ref(max, t))))
      ArrayFold(a, NA(t), max, value, body)
    }

    registerIR("[]", TArray(tv("T")), TInt32()) { (a, i) => ArrayRef(a, i) }

    registerIR("[:]", TArray(tv("T"))) { (a) => a }

    registerIR("[*:]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
          ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
          i),
          ArrayLen(a),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("[:*]", TArray(tv("T")), TInt32()) { (a, i) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(I32(0),
          If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
            i),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }

    registerIR("[*:*]", TArray(tv("T")), TInt32(), TInt32()) { (a, i, j) =>
      val idx = genUID()
      ArrayMap(
        ArrayRange(
          If(ApplyComparisonOp(LT(TInt32()), i, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), i),
            i),
          If(ApplyComparisonOp(LT(TInt32()), j, I32(0)),
            ApplyBinaryPrimOp(Add(), ArrayLen(a), j),
            j),
          I32(1)),
        idx,
        ArrayRef(a, Ref(idx, TInt32())))
    }
  }
}
