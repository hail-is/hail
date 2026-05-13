package is.hail.expr.ir.functions

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.defs._
import is.hail.types.virtual._

object SetFunctions extends RegistryFunctions {
  def contains(set: Atom, elem: Atom): IR =
    guardIR(!IsNA(set)) {
      M.eval {
        for {
          i <- LowerBoundOnOrderedCollection(set, elem, onKey = false)
          a <- CastToArray(set)
        } yield (i < ArrayLen(a)) && (ArrayRef(a, i) ceq elem)
      }
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
      ToSet(
        flatten(MakeStream(
          ArraySeq(CastToArray(s), MakeArray(v)),
          TStream(TArray(v.typ)),
        ))
      )
    }

    registerIR2("union", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      val t = s1.typ.asInstanceOf[TSet].elementType
      ToSet(flatten(MakeStream(ArraySeq(CastToArray(s1), CastToArray(s2)), TStream(TArray(t)))))
    }

    registerIR2("intersection", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      ToSet(filterIR(ToStream(s1))(contains(s2, _)))
    }

    registerIR2("difference", TSet(tv("T")), TSet(tv("T")), TSet(tv("T"))) { (_, s1, s2, _) =>
      ToSet(filterIR(ToStream(s1))(!contains(s2, _)))
    }

    registerIR2("isSubset", TSet(tv("T")), TSet(tv("T")), TBoolean) { (_, s, w, _) =>
      guardIR(!(IsNA(s) || IsNA(w))) {
        M.eval {
          for {
            sArray <- CastToArray(s)
            sLen <- ArrayLen(sArray)
            wArray <- CastToArray(w)
            wLen <- ArrayLen(wArray)
          } yield (sLen <= wLen) && tailLoop(TBoolean, 0) { case (recur, Seq(i)) =>
            val isElem =
              M.eval {
                for {
                  elem <- ArrayRef(sArray, i)
                  j <- LowerBoundOnOrderedCollection(wArray, elem, onKey = false)
                } yield (j < wLen) && (ArrayRef(wArray, j) ceq elem)
              }

            If(i >= sLen, True(), If(isElem, recur(ArraySeq(i + 1)), False()))
          }
        }
      }
    }

    registerIR1("median", TSet(tnum("T")), tv("T")) { (_, s, _) =>
      val t = s.typ.asInstanceOf[TSet].elementType
      def div(a: IR, b: IR): IR = ApplyBinaryPrimOp(BinaryOp.defaultDivideOp(t), a, b)

      bindIR(CastToArray(s)) { a =>
        bindIR(ArrayLen(a)) { len =>
          def ref(i: IR) = ArrayRef(a, i)
          If(
            IsNA(a) || (len ceq 0),
            NA(t),
            bindIR(len - 1) { lastIdx =>
              bindIR(If(IsNA(ref(lastIdx)), lastIdx, len)) { size =>
                If(
                  size ceq 0,
                  NA(t),
                  bindIR((size - 1) floorDiv 2) { midIdx =>
                    If(
                      invoke("mod", TInt32, size, 2) cne 0,
                      ref(midIdx), // odd number of non-missing elements
                      div(ref(midIdx) + ref(midIdx + 1), Cast(2, t)),
                    )
                  },
                )
              }
            },
          )
        }
      }
    }
  }
}
