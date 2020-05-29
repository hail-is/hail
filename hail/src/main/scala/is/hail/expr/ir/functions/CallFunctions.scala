package is.hail.expr.ir.functions

import is.hail.expr.ir.InferPType
import is.hail.types._
import is.hail.types.physical.{PBoolean, PCanonicalArray, PCanonicalCall, PInt32, PType}
import is.hail.types.virtual._
import is.hail.utils.FastSeq
import is.hail.variant._

object CallFunctions extends RegistryFunctions {
  def registerAll() {
    registerWrappedScalaFunction1("Call", TString, TCall, (rt: Type, st: PType) => PCanonicalCall(st.required))(Call.getClass, "parse")

    registerScalaFunction("Call", Array(TBoolean), TCall, (rt: Type, _: Seq[PType]) => PCanonicalCall())(Call0.getClass, "apply")

    registerScalaFunction("Call", Array(TInt32, TBoolean), TCall, (rt: Type, _: Seq[PType]) => PCanonicalCall())(Call1.getClass, "apply")

    registerScalaFunction("Call", Array(TInt32, TInt32, TBoolean), TCall, (rt: Type, _: Seq[PType]) => PCanonicalCall())(Call2.getClass, "apply")

    registerScalaFunction("UnphasedDiploidGtIndexCall", Array(TInt32), TCall, (rt: Type, _: Seq[PType]) => PCanonicalCall())(Call2.getClass, "fromUnphasedDiploidGtIndex")

    registerWrappedScalaFunction2("Call", TArray(TInt32), TBoolean, TCall, {
      case(rt: Type, _: PType, _: PType) => PCanonicalCall()
    })(CallN.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities) registerScalaFunction(q, Array(TCall), TBoolean, (rt: Type, _: Seq[PType]) => PBoolean())(Call.getClass, q)

    registerScalaFunction("ploidy", Array(TCall), TInt32, (rt: Type, _: Seq[PType]) => PInt32())(Call.getClass, "ploidy")

    registerScalaFunction("nNonRefAlleles", Array(TCall), TInt32, (rt: Type, _: Seq[PType]) => PInt32())(Call.getClass, "nNonRefAlleles")

    registerScalaFunction("unphasedDiploidGtIndex", Array(TCall), TInt32, (rt: Type, _: Seq[PType]) => PInt32())(Call.getClass, "unphasedDiploidGtIndex")

    registerScalaFunction("index", Array(TCall, TInt32), TInt32, (rt: Type, _: Seq[PType]) => PInt32())(Call.getClass, "alleleByIndex")

    registerScalaFunction("downcode", Array(TCall, TInt32), TCall, (rt: Type, _: Seq[PType]) => PCanonicalCall())(Call.getClass, "downcode")

    registerWrappedScalaFunction2("oneHotAlleles", TCall, TInt32, TArray(TInt32), {
      case(rt: Type, _: PType, _: PType) => PCanonicalArray(PInt32(true))
    })(Call.getClass, "oneHotAlleles")
  }
}
