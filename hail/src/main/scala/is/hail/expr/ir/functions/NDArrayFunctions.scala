package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types.coerce
import is.hail.expr.types.virtual._

object NDArrayFunctions extends RegistryFunctions {
  val ndArrayOps: Array[(String, Type, (IR, IR) => IR)] =
    Array(
      ("*", tnum("T"), ApplyBinaryPrimOp(Multiply(), _, _)),
      ("/", TInt32(), ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("/", TInt64(), ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("/", TFloat32(), ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("/", TFloat64(), ApplyBinaryPrimOp(FloatingPointDivide(), _, _)),
      ("//", tnum("T"), ApplyBinaryPrimOp(RoundToNegInfDivide(), _, _)),
      ("+", tnum("T"), ApplyBinaryPrimOp(Add(), _, _)),
      ("-", tnum("T"), ApplyBinaryPrimOp(Subtract(), _, _)),
      ("**", tnum("T"), (ir1: IR, ir2: IR) => Apply("**", Seq(ir1, ir2))),
      ("%", tnum("T"), (ir1: IR, ir2: IR) => Apply("%", Seq(ir1, ir2))))

  override def registerAll() {
    for ((stringOp, argType, irOp) <- ndArrayOps) {
      registerIR(stringOp, tv("T", "ndarray"), argType, tv("T", "ndarray")) { (a, c) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(Ref(i, c.typ), c))
      }

      registerIR(stringOp, argType, tv("T", "ndarray"), tv("T", "ndarray")) { (c, a) =>
        val i = genUID()
        NDArrayMap(a, i, irOp(c, Ref(i, c.typ)))
      }

      registerIR(stringOp, tv("T", "ndarray"), tv("T", "ndarray"), tv("T", "ndarray")) { (l, r) =>
        val lid = genUID()
        val rid = genUID()
        val lElemRef = Ref(lid, coerce[TNDArray](l.typ).elementType)
        val rElemRef = Ref(rid, coerce[TNDArray](r.typ).elementType)

        NDArrayMap2(l, r, lid, rid, irOp(lElemRef, rElemRef))
      }
    }
  }
}
