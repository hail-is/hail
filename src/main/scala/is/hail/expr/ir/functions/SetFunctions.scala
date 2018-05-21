package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils.FastSeq

object SetFunctions extends RegistryFunctions {
  def registerAll() {
    registerIR("toSet", TArray(tv("T"))) { a =>
      ToSet(a)
    }

    registerIR("size", TSet(tv("T"))) { s =>
      ArrayLen(ToArray(s))
    }

    registerIR("isEmpty", TSet(tv("T"))) { s =>
      ArrayFunctions.isEmpty(ToArray(s))
    }

    registerIR("contains", TSet(tv("T")), tv("T")) { (s, v) =>
      SetContains(s, v)
    }

    registerIR("remove", TSet(tv("T")), tv("T")) { (s, v) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        ArrayFilter(
          ToArray(s),
          x,
          ApplyComparisonOp(NEQWithNA(t), Ref(x, t), v)))
    }

    registerIR("add", TSet(tv("T")), tv("T")) { (s, v) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        ArrayFlatMap(
          MakeArray(FastSeq(ToArray(s), MakeArray(FastSeq(v), TArray(t))), TArray(TArray(t))),
          x,
          Ref(x, TArray(t))))
    }

    registerIR("union", TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFlatMap(
          MakeArray(FastSeq(ToArray(s1), ToArray(s2)), TArray(TArray(t))),
          x,
          Ref(x, TArray(t))))
    }

    registerIR("intersection", TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          SetContains(s2, Ref(x, t))))
    }

    registerIR("difference", TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          ApplyUnaryPrimOp(Bang(), SetContains(s2, Ref(x, t)))))
    }

    registerIR("isSubset", TSet(tv("T")), TSet(tv("T"))) { (s, w) =>
      val t = -s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val x = genUID()
      ArrayFold(ToArray(s), True(), a, x,
        // FIXME short circuit
        ApplySpecial("&&",
          FastSeq(Ref(a, TBoolean()), SetContains(w, Ref(x, t)))))
    }

    registerIR("sum", TSet(tnum("T"))) { s =>
      ArrayFunctions.sum(ToArray(s))
    }

    registerIR("product", TSet(tnum("T"))) { s =>
      ArrayFunctions.product(ToArray(s))
    }

    registerIR("min", TSet(tnum("T"))) { s =>
      val t = s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      Let(a, ToArray(s), If(
        ApplyComparisonOp(GT(TInt32()), ArrayLen(Ref(a, TArray(t))), I32(0)),
        ArrayRef(Ref(a, TArray(t)), I32(0)),
        NA(t)))
    }

    registerIR("max", TSet(tnum("T"))) { s =>
      val t = s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val size = genUID()
      val last = genUID()

      Let(a, ToArray(s),
        Let(size, ArrayLen(Ref(a, TArray(t))),
          If(ApplyComparisonOp(EQ(TInt32()), Ref(size, TInt32()), I32(0)),
            NA(t),
            Let(last, ArrayRef(Ref(a, TArray(t)), ApplyBinaryPrimOp(Subtract(), Ref(size, TInt32()), I32(1))),
              If(IsNA(Ref(last, t)),
                If(ApplyComparisonOp(EQ(TInt32()), Ref(size, TInt32()), I32(1)),
                  NA(t),
                  ArrayRef(Ref(a, TArray(t)), ApplyBinaryPrimOp(Subtract(), Ref(size, TInt32()), I32(2)))),
                Ref(last, t))))))
    }

    registerIR("mean", TSet(tnum("T"))) { s => ArrayFunctions.mean(ToArray(s)) }

    registerIR("median", TSet(tnum("T"))) { s =>
      val t = s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val size = genUID()
      val lastIdx = genUID()
      val midIdx = genUID()
      val midIdx2 = genUID()

      Let(a, ToArray(s),
        Let(size, ArrayLen(Ref(a, TArray(t))),
          If(ApplyComparisonOp(EQ(TInt32()), Ref(size, TInt32()), I32(0)),
            NA(t),
            If(ApplyComparisonOp(EQ(TInt32()), Ref(size, TInt32()), I32(1)),
              ArrayRef(Ref(a, TArray(t)), I32(0)),
              Let(lastIdx, ApplyBinaryPrimOp(Subtract(), Ref(size, TInt32()), I32(1)),
                Let(lastIdx, If(
                  IsNA(ArrayRef(Ref(a, TArray(t)), Ref(lastIdx, TInt32()))),
                  ApplyBinaryPrimOp(Subtract(), Ref(lastIdx, TInt32()), I32(1)),
                  Ref(lastIdx, TInt32())),
                  Let(midIdx, ApplyBinaryPrimOp(RoundToNegInfDivide(), Ref(lastIdx, TInt32()), I32(2)),
                    If(ApplyComparisonOp(EQ(TInt32()), Apply("%", FastSeq(Ref(lastIdx, TInt32()), I32(2))), I32(0)),
                      ArrayRef(Ref(a, TArray(t)), Ref(midIdx, TInt32())), // odd number of non-missing elements
                      Let(midIdx2, ApplyBinaryPrimOp(Add(), Ref(midIdx, TInt32()), I32(1)), // even number of non-missing elements
                        ApplyBinaryPrimOp(
                          RoundToNegInfDivide(),
                          ApplyBinaryPrimOp(
                            Add(),
                            ArrayRef(Ref(a, TArray(t)), Ref(midIdx, TInt32())),
                            ArrayRef(Ref(a, TArray(t)), Ref(midIdx2, TInt32()))),
                          Cast(I32(2), t)))))))))))
    }
  }
}
