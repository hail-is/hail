package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types.{TArray, TBoolean, TSet}
import is.hail.utils.FastSeq

object SetFunctions extends RegistryFunctions {
  def registerAll() {
    registerIR("toSet", TArray(tv("T"))) { a =>
      ToSet(a)
    }

    registerIR("contains", TSet(tv("T")), tv("T")) { case (s, v) =>
      val t = v.typ
      val x = genUID()
      val accum = genUID()

      ArrayFold(ToArray(s), False(), accum, x,
        ApplySpecial("||",
          FastSeq(Ref(accum, TBoolean()), ApplyBinaryPrimOp(EQ(), v, Ref(x, t)))))
    }

    registerIR("remove", TSet(tv("T")), tv("T")) { case (s, v) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        ArrayFilter(
          ToArray(s),
          x,
          ApplyBinaryPrimOp(NEQ(), Ref(x, t), v)))
    }

    registerIR("add", TSet(tv("T")), tv("T")) { case (s, v) =>
      val t = v.typ
      val x = genUID()
      ToSet(
        ArrayFlatMap(
          MakeArray(FastSeq(ToArray(s), MakeArray(FastSeq(v), TArray(t))), TArray(TArray(t))),
          x,
          Ref(x, TArray(t))))
    }

    registerIR("union", TSet(tv("T")), TSet(tv("T"))) { case (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFlatMap(
          MakeArray(FastSeq(ToArray(s1), ToArray(s2)), TArray(TArray(t))),
          x,
          Ref(x, TArray(t))))
    }

    registerIR("intersection", TSet(tv("T")), TSet(tv("T"))) { case (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          IRFunctionRegistry.invoke("contains", FastSeq(s2, Ref(x, t)))))
    }

    registerIR("difference", TSet(tv("T")), TSet(tv("T"))) { case (s1, s2) =>
      val t = -s1.typ.asInstanceOf[TSet].elementType
      val x = genUID()
      ToSet(
        ArrayFilter(ToArray(s1), x,
          ApplyUnaryPrimOp(Bang(), IRFunctionRegistry.invoke("contains", FastSeq(s2, Ref(x, t))))))
    }

    registerIR("isSubset", TSet(tv("T")), TSet(tv("T"))) { case (s, w) =>
      val t = -s.typ.asInstanceOf[TSet].elementType
      val a = genUID()
      val x = genUID()
      ArrayFold(ToArray(s), True(), a, x,
        // FIXME short circuit
        ApplySpecial("&&",
          FastSeq(Ref(a, TBoolean()), IRFunctionRegistry.invoke("contains", FastSeq(w, Ref(x, t))))))
    }
  }
}
