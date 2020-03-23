package is.hail.expr.ir.functions

import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.utils.FastSeq
import is.hail.variant._

object CallFunctions extends RegistryFunctions {
  def registerAll() {
    registerWrappedScalaFunction("Call", TString, TCall, null)(Call.getClass, "parse")

    registerScalaFunction("Call", Array(TBoolean), TCall, null)(Call0.getClass, "apply")

    registerScalaFunction("Call", Array(TInt32, TBoolean), TCall, null)(Call1.getClass, "apply")

    registerScalaFunction("Call", Array(TInt32, TInt32, TBoolean), TCall, null)(Call2.getClass, "apply")

    registerScalaFunction("UnphasedDiploidGtIndexCall", Array(TInt32), TCall, null)(Call2.getClass, "fromUnphasedDiploidGtIndex")

    registerWrappedScalaFunction("Call", TArray(TInt32), TBoolean, TCall, null)(CallN.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities) registerScalaFunction(q, Array(TCall), TBoolean, null)(Call.getClass, q)

    registerScalaFunction("ploidy", Array(TCall), TInt32, null)(Call.getClass, "ploidy")

    registerScalaFunction("nNonRefAlleles", Array(TCall), TInt32, null)(Call.getClass, "nNonRefAlleles")

    registerScalaFunction("unphasedDiploidGtIndex", Array(TCall), TInt32, null)(Call.getClass, "unphasedDiploidGtIndex")

    registerScalaFunction("index", Array(TCall, TInt32), TInt32, null)(Call.getClass, "alleleByIndex")

    registerScalaFunction("downcode", Array(TCall, TInt32), TCall, null)(Call.getClass, "downcode")

    registerWrappedScalaFunction("oneHotAlleles", TCall, TInt32, TArray(TInt32), null)(Call.getClass, "oneHotAlleles")
  }
}
