package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types
import is.hail.utils.FastSeq

object SetFunctions extends RegistryFunctions {
  def contains(set: IR, elem: IR) =
    If(IsNA(set),
      NA(TBoolean()),
      ApplyComparisonOp(
        EQWithNA(elem.typ),
        ArrayRef(ToArray(set), LowerBoundOnOrderedCollection(set, elem, onKey=false)),
        elem))

  def registerAll() {
    registerIR("toSet", TArray(tv("T"))) { a =>
      ToSet(a)
    }

    registerIR("toArray", TSet(tv("T"))) { a =>
      ToArray(a)
    }

    registerIR("size", TSet(tv("T"))) { s =>
      ArrayLen(ToArray(s))
    }

    registerIR("isEmpty", TSet(tv("T"))) { s =>
      ArrayFunctions.isEmpty(ToArray(s))
    }

    registerIR("contains", TSet(tv("T")), tv("T"))(contains)

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
          contains(s2, Ref(x, t))))
    }

    registerIR("difference", TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          ApplyUnaryPrimOp(Bang(), contains(s2, Ref(x, t)))))
    }

    registerIR("isSubset", TSet(tv("T")), TSet(tv("T"))) { (s, w) =>
      val t = -s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val x = genUID()
      ArrayFold(ToArray(s), True(), a, x,
        // FIXME short circuit
        ApplySpecial("&&",
          FastSeq(Ref(a, TBoolean()), contains(w, Ref(x, t)))))
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
      val t = -s.typ.asInstanceOf[TSet].elementType
      val a = Ref(genUID(), TArray(t))
      val size = Ref(genUID(), TInt32())
      val lastIdx = size - 1
      val midIdx = lastIdx.floorDiv(2)
      def ref(i: IR) = ArrayRef(a, i)
      val len: IR = ArrayLen(a)
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      Let(a.name, ToArray(s),
        If(IsNA(a),
          NA(t),
          Let(size.name,
            If(len.ceq(0), len, If(IsNA(ref(len - 1)), len - 1, len)),
            If(size.ceq(0),
              NA(t),
              If(invoke("%", size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)))))))
    }

    registerIR("flatten", TSet(tv("T"))) { s =>
      val elt = Ref(genUID(), types.coerce[TContainer](s.typ).elementType)
      ToSet(ArrayFlatMap(ToArray(s), elt.name, ToArray(elt)))
    }
  }
}
