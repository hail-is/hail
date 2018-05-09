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

    registerIR("isSubset", TSet(tv("T")), TSet(tv("T"))) { case (s, w) =>
      val t = -s.typ.asInstanceOf[TSet].elementType

      val a = genUID()
      val x = genUID()

      val args = FastSeq(w, Ref(x, t))
      println(args, args.map(_.typ))

      ArrayFold(ToArray(s), True(), a, x,
        // FIXME short circuit
        ApplySpecial("&&",
          FastSeq(Ref(a, TBoolean()), Apply("contains", args))))
    }

    // union(set<T>,set<T>):set<T>
    // intersection(set<T>,set<T>):set<T>
    // difference(set<T>,set<T>):set<T>
  }
}
