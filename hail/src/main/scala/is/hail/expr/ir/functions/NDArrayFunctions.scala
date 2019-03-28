package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types.coerce
import is.hail.expr.types.virtual._

object NDArrayFunctions extends RegistryFunctions {
  override def registerAll() {
    for ((stringOp, argType, retType, irOp) <- ArrayFunctions.arrayOps) {
      registerIR(stringOp, TNDArray(argType, tv("U", "nat")), argType,
        TNDArray(retType, tv("U"))) { (a, c) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR(stringOp, argType, TNDArray(argType, tv("U", "nat")),
        TNDArray(retType, tv("U"))) { (a, c) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR(stringOp, TNDArray(argType, tv("U", "nat")), TNDArray(argType, tv("U")),
        TNDArray(argType, tv("U"))) { (l, r) =>
        val lid = genUID()
        val rid = genUID()
        val lElemRef = Ref(lid, coerce[TNDArray](l.typ).elementType)
        val rElemRef = Ref(rid, coerce[TNDArray](r.typ).elementType)

        NDArrayMap2(l, r, lid, rid, irOp(lElemRef, rElemRef))
      }
    }
  }
}
