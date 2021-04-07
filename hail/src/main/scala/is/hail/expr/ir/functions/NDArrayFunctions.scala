package is.hail.expr.ir.functions

import is.hail.expr.{Nat, NatVariable}
import is.hail.expr.ir._
import is.hail.types.coerce
import is.hail.types.virtual._

object NDArrayFunctions extends RegistryFunctions {
  override def registerAll() {
    for ((stringOp, argType, retType, irOp) <- ArrayFunctions.arrayOps) {
      val nDimVar = NatVariable()
      registerIR2(stringOp, TNDArray(argType, nDimVar), argType, TNDArray(retType, nDimVar)) { (_, a, c) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR2(stringOp, argType, TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (_, c, a) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR2(stringOp, TNDArray(argType, nDimVar), TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (_, l, r) =>
        val lid = genUID()
        val rid = genUID()
        val lElemRef = Ref(lid, coerce[TNDArray](l.typ).elementType)
        val rElemRef = Ref(rid, coerce[TNDArray](r.typ).elementType)

        NDArrayMap2(l, r, lid, rid, irOp(lElemRef, rElemRef))
      }
    }

    registerIEmitCode2("linear_solve", TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), TNDArray(TFloat64, Nat(2)), { (t, p1, p2) => p2 }) { case (cb, r, pt, aec, bec) =>
      aec.toI(cb).flatMap(cb){ apc =>
        bec.toI(cb).map(cb){ bpc =>
          val a = apc.asNDArray.memoize(cb, "A")
          val b = bpc.asNDArray.memoize(cb, "B")
          val IndexedSeq(m, k) = b.shapes(cb)
        }
      }
      ???
    }
  }
}
