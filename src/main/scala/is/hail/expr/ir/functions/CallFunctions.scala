package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types._
import is.hail.variant.{Call, Call2}

object CallFunctions extends RegistryFunctions {

  def registerAll() {
    registerScalaFunction("downcode", TCall(), TInt32(), TCall())(Call.getClass, "downcode")
    registerScalaFunction("UnphasedDiploidGtIndexCall", TInt32(), TCall())(Call2.getClass, "fromUnphasedDiploidGtIndex")
    registerScalaFunction("Call", TInt32(), TInt32(), TBoolean(), TCall())(Call2.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities) registerScalaFunction(q, TCall(), TBoolean())(Call.getClass, q)

    registerScalaFunction("nNonRefAlleles", TCall(), TInt32())(Call.getClass, "nNonRefAlleles")
  }
}
