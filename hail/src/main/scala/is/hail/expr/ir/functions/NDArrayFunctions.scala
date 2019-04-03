package is.hail.expr.ir.functions

import is.hail.expr.NatVariable
import is.hail.expr.ir._
import is.hail.expr.types.coerce
import is.hail.expr.types.virtual._

object NDArrayFunctions extends RegistryFunctions {
  override def registerAll() {
    for ((stringOp, argType, retType, irOp) <- ArrayFunctions.arrayOps) {
      val nDimVar = NatVariable()
      registerIR(stringOp, TNDArray(argType, nDimVar), argType, TNDArray(retType, nDimVar)) { (a, c) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR(stringOp, argType, TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (c, a) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR(stringOp, TNDArray(argType, nDimVar), TNDArray(argType, nDimVar), TNDArray(retType, nDimVar)) { (l, r) =>
        val lid = genUID()
        val rid = genUID()
        val lElemRef = Ref(lid, coerce[TNDArray](l.typ).elementType)
        val rElemRef = Ref(rid, coerce[TNDArray](r.typ).elementType)

        NDArrayMap2(l, r, lid, rid, irOp(lElemRef, rElemRef))
      }
    }
  }
}
