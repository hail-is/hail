package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

object SetFunctions extends RegistryFunctions {
  def contains(set: IR, elem: IR) = {
    val i = Ref(genUID(), TInt32)

    If(IsNA(set),
      NA(TBoolean),
      Let(i.name,
        LowerBoundOnOrderedCollection(set, elem, onKey = false),
        If(i.ceq(ArrayLen(CastToArray(set))),
          False(),
          ApplyComparisonOp(EQWithNA(elem.typ), ArrayRef(CastToArray(set), i), elem))))
  }

  def registerAll() {
    registerIR1("toSet", TArray(tv("T")), TSet(tv("T"))) { (_, a, _) =>
      ToSet(ToStream(a))
    }

    registerIR1("isEmpty", TSet(tv("T")), TBoolean) { (_, s, _) =>
      ArrayFunctions.isEmpty(CastToArray(s))
    }

    registerIR2("contains", TSet(tv("T")), tv("T"), TBoolean)((_, a, b, _) => contains(a, b))

    registerIR2("remove", TSet(tv("T")), tv("T"), TSet(tv("T"))) { (_, s, v, _) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        StreamFilter(
          ToStream(s),
          x,
          ApplyComparisonOp(NEQWithNA(t), Ref(x, t), v)))
    }

    registerIR2("add", TSet(tv("T")), tv("T"), TSet(tv("T"))) { (_, s, v, _) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        StreamFlatMap(
          MakeStream(FastSeq(CastToArray(s), MakeArray(FastSeq(v), TArray(t))), TStream(TArray(t))),
          x,
          ToStream(Ref(x, TArray(t)))))
    }

    registerIR2("union", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      val t = s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        StreamFlatMap(
          MakeStream(FastSeq(CastToArray(s1), CastToArray(s2)), TStream(TArray(t))),
          x,
          ToStream(Ref(x, TArray(t)))))
    }

    registerIR2("intersection", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      val t = s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        StreamFilter(ToStream(s1), x,
          contains(s2, Ref(x, t))))
    }

    registerIR2("difference", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      val t = s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        StreamFilter(ToStream(s1), x,
          ApplyUnaryPrimOp(Bang, contains(s2, Ref(x, t)))))
    }

    registerIR2("isSubset", TSet(tv("T")), TSet(tv("T")), TBoolean) { (_, s, w, errorID) =>
      val t = s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val x = genUID()
      StreamFold(ToStream(s), True(), a, x,
        // FIXME short circuit
        ApplySpecial("land", FastSeq(), FastSeq(Ref(a, TBoolean), contains(w, Ref(x, t))), TBoolean, errorID))
    }

    registerIR1("median", TSet(tnum("T")), tv("T")) { (_, s, _) =>
      val t = s.typ.asInstanceOf[TSet].elementType
      val a = Ref(genUID(), TArray(t))
      val size = Ref(genUID(), TInt32)
      val lastIdx = size - 1
      val midIdx = lastIdx.floorDiv(2)
      def ref(i: IR) = ArrayRef(a, i)
      val len: IR = ArrayLen(a)
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      Let(a.name, CastToArray(s),
        If(IsNA(a),
          NA(t),
          Let(size.name,
            If(len.ceq(0), len, If(IsNA(ref(len - 1)), len - 1, len)),
            If(size.ceq(0),
              NA(t),
              If(invoke("mod", TInt32, size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)))))))
    }
  }
}
