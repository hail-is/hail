package is.hail.expr.ir.functions

import is.hail.expr.types._
import is.hail.variant._

object CallFunctions extends RegistryFunctions {
  def registerAll() {
    registerWrappedScalaFunction("Call", TString(), TCall())(Call.getClass, "parse")

    registerScalaFunction("Call", TBoolean(), TCall())(Call0.getClass, "apply")

    registerScalaFunction("Call", TInt32(), TBoolean(), TCall())(Call1.getClass, "apply")

    registerScalaFunction("Call", TInt32(), TInt32(), TBoolean(), TCall())(Call2.getClass, "apply")

    registerScalaFunction("UnphasedDiploidGtIndexCall", TInt32(), TCall())(Call2.getClass, "fromUnphasedDiploidGtIndex")

    registerWrappedScalaFunction("Call", TArray(TInt32()), TBoolean(), TCall())(CallN.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities) registerScalaFunction(q, TCall(), TBoolean())(Call.getClass, q)

    registerScalaFunction("ploidy", TCall(), TInt32())(Call.getClass, "ploidy")

    registerScalaFunction("nNonRefAlleles", TCall(), TInt32())(Call.getClass, "nNonRefAlleles")

    registerScalaFunction("unphasedDiploidGtIndex", TCall(), TInt32())(Call.getClass, "unphasedDiploidGtIndex")

    registerScalaFunction("[]", TCall(), TInt32(), TInt32())(Call.getClass, "alleleByIndex")

    registerScalaFunction("downcode", TCall(), TInt32(), TCall())(Call.getClass, "downcode")

    registerWrappedScalaFunction("oneHotAlleles", TCall(), TInt32(), TArray(TInt32()))(Call.getClass, "oneHotAlleles")
  }
}
