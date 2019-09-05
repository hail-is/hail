package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types.virtual._
import is.hail.utils.FastSeq

object SetFunctions extends RegistryFunctions {
  def contains(set: IR, elem: IR) = {
    val i = Ref(genUID(), TInt32())

    If(IsNA(set),
      NA(TBoolean()),
      Let(i.name,
        LowerBoundOnOrderedCollection(set, elem, onKey = false),
        If(i.ceq(ArrayLen(ToArray(set))),
          False(),
          ApplyComparisonOp(EQWithNA(elem.typ), ArrayRef(ToArray(set), i), elem))))
  }

  def registerAll() {
    registerIR("toSet", TArray(tv("T")), TSet(tv("T"))) { a =>
      ToSet(a)
    }

    registerIR("isEmpty", TSet(tv("T")), TBoolean()) { s =>
      ArrayFunctions.isEmpty(ToArray(s))
    }

    registerIR("contains", TSet(tv("T")), tv("T"), TBoolean())(contains)

    registerIR("remove", TSet(tv("T")), tv("T"), TSet(tv("T"))) { (s, v) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        ArrayFilter(
          ToArray(s),
          x,
          ApplyComparisonOp(NEQWithNA(t), Ref(x, t), v)))
    }

    registerIR("add", TSet(tv("T")), tv("T"), TSet(tv("T"))) { (s, v) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        ArrayFlatMap(
          MakeArray(FastSeq(ToArray(s), MakeArray(FastSeq(v), TArray(t))), TArray(TArray(t))),
          x,
          Ref(x, TArray(t))))
    }

    registerIR("union", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFlatMap(
          MakeArray(FastSeq(ToArray(s1), ToArray(s2)), TArray(TArray(t))),
          x,
          Ref(x, TArray(t))))
    }

    registerIR("intersection", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          contains(s2, Ref(x, t))))
    }

    registerIR("difference", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          ApplyUnaryPrimOp(Bang(), contains(s2, Ref(x, t)))))
    }

    registerIR("isSubset", TSet(tv("T")), TSet(tv("T")), TBoolean()) { (s, w) =>
      val t = -s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val x = genUID()
      ArrayFold(ToArray(s), True(), a, x,
        // FIXME short circuit
        ApplySpecial("&&",
          FastSeq(Ref(a, TBoolean()), contains(w, Ref(x, t))), TBoolean()))
    }

    registerIR("sum", TSet(tnum("T")), tv("T")) { s =>
      ArrayFunctions.sum(ToArray(s))
    }

    registerIR("product", TSet(tnum("T")), tv("T")) { s =>
      ArrayFunctions.product(ToArray(s))
    }

    registerIR("mean", TSet(tnum("T")), TFloat64()) { s => ArrayFunctions.mean(ToArray(s)) }

    registerIR("median", TSet(tnum("T")), tv("T")) { s =>
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
              If(invoke("%", TInt32(), size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)))))))
    }
  }
}
