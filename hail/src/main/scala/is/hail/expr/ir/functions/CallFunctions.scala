package is.hail.expr.ir.functions

import is.hail.expr.ir.InferPType
import is.hail.expr.types._
import is.hail.expr.types.physical.{PBoolean, PCanonicalArray, PCanonicalCall, PInt32, PType}
import is.hail.expr.types.virtual._
import is.hail.utils.FastSeq
import is.hail.variant._

object CallFunctions extends RegistryFunctions {
  def registerAll() {
    registerWrappedScalaFunction("Call", TString, TCall, (_: PType) => PCanonicalCall())(Call.getClass, "parse")

    registerScalaFunction("Call", Array(TBoolean), TCall, (pt: Seq[PType]) => PCanonicalCall())(Call0.getClass, "apply")

    registerScalaFunction("Call", Array(TInt32, TBoolean), TCall, (pt: Seq[PType]) => PCanonicalCall())(Call1.getClass, "apply")

    registerScalaFunction("Call", Array(TInt32, TInt32, TBoolean), TCall, (pt: Seq[PType]) => PCanonicalCall())(Call2.getClass, "apply")

    registerScalaFunction("UnphasedDiploidGtIndexCall", Array(TInt32), TCall, (pt: Seq[PType]) => PCanonicalCall())(Call2.getClass, "fromUnphasedDiploidGtIndex")

    registerWrappedScalaFunction("Call", TArray(TInt32), TBoolean, TCall, {
      case(_: PType, _: PType) => PCanonicalCall()
    })(CallN.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities) registerScalaFunction(q, Array(TCall), TBoolean, (pt: Seq[PType]) => PBoolean())(Call.getClass, q)

    registerScalaFunction("ploidy", Array(TCall), TInt32, (pt: Seq[PType]) => PInt32())(Call.getClass, "ploidy")

    registerScalaFunction("nNonRefAlleles", Array(TCall), TInt32, (pt: Seq[PType]) => PInt32())(Call.getClass, "nNonRefAlleles")

    registerScalaFunction("unphasedDiploidGtIndex", Array(TCall), TInt32, (pt: Seq[PType]) => PInt32())(Call.getClass, "unphasedDiploidGtIndex")

    registerScalaFunction("index", Array(TCall, TInt32), TInt32, (pt: Seq[PType]) => PInt32())(Call.getClass, "alleleByIndex")

    registerScalaFunction("downcode", Array(TCall, TInt32), TCall, (pt: Seq[PType]) => PCanonicalCall())(Call.getClass, "downcode")

    registerWrappedScalaFunction("oneHotAlleles", TCall, TInt32, TArray(TInt32), {
      case(_: PType, _: PType) => PCanonicalArray(PInt32(true))
    })(Call.getClass, "oneHotAlleles")
  }
}
