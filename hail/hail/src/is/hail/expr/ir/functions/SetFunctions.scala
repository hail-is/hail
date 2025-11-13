package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

object SetFunctions extends RegistryFunctions {
  def contains(set: IR, elem: IR) = {
    If(
      IsNA(set),
      NA(TBoolean),
      bindIR(LowerBoundOnOrderedCollection(set, elem, onKey = false)) { i =>
        If(
          i.ceq(ArrayLen(CastToArray(set))),
          False(),
          ApplyComparisonOp(EQWithNA, ArrayRef(CastToArray(set), i), elem),
        )
      },
    )
  }

  override def registerAll(): Unit = {
    registerIR1("toSet", TArray(tv("T")), TSet(tv("T")))((_, a, _) => ToSet(ToStream(a)))

    registerIR1("isEmpty", TSet(tv("T")), TBoolean) { (_, s, _) =>
      ArrayFunctions.isEmpty(CastToArray(s))
    }

    registerIR2("contains", TSet(tv("T")), tv("T"), TBoolean)((_, a, b, _) => contains(a, b))

    registerIR2("remove", TSet(tv("T")), tv("T"), TSet(tv("T"))) { (_, s, v, _) =>
      ToSet(filterIR(ToStream(s))(ApplyComparisonOp(NEQWithNA, _, v)))
    }

    registerIR2("add", TSet(tv("T")), tv("T"), TSet(tv("T"))) { (_, s, v, _) =>
      val t = v.typ
      ToSet(
        flatMapIR(MakeStream(
          FastSeq(CastToArray(s), MakeArray(FastSeq(v), TArray(t))),
          TStream(TArray(t)),
        ))(ToStream(_))
      )
    }

    registerIR2("union", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      val t = s1.typ.asInstanceOf[TSet].elementType
      ToSet(
        flatMapIR(
          MakeStream(FastSeq(CastToArray(s1), CastToArray(s2)), TStream(TArray(t)))
        )(ToStream(_))
      )
    }

    registerIR2("intersection", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      ToSet(filterIR(ToStream(s1))(contains(s2, _)))
    }

    registerIR2("difference", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      ToSet(
        filterIR(ToStream(s1))(x => ApplyUnaryPrimOp(Bang, contains(s2, x)))
      )
    }

    registerIR2("isSubset", TSet(tv("T")), TSet(tv("T")), TBoolean) { (_, s, w, errorID) =>
      foldIR(ToStream(s), True()) { (a, x) =>
        // FIXME short circuit
        ApplySpecial(
          "land",
          FastSeq(),
          FastSeq(a, contains(w, x)),
          TBoolean,
          errorID,
        )
      }
    }

    registerIR1("median", TSet(tnum("T")), tv("T")) { (_, s, _) =>
      val t = s.typ.asInstanceOf[TSet].elementType
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      bindIR(CastToArray(s)) { a =>
        def ref(i: IR) = ArrayRef(a, i)
        def len: IR = ArrayLen(a)
        If(
          IsNA(a),
          NA(t),
          bindIR(If(len.ceq(0), len, If(IsNA(ref(len - 1)), len - 1, len))) { size =>
            val lastIdx = size - 1
            val midIdx = lastIdx.floorDiv(2)
            If(
              size.ceq(0),
              NA(t),
              If(
                invoke("mod", TInt32, size, 2).cne(0),
                ref(midIdx), // odd number of non-missing elements
                div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)),
              ),
            )
          },
        )
      }
    }
  }
}
