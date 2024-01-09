package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.types.physical.{PCanonicalArray, PInt32}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SIndexablePointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SInt32}
import is.hail.types.virtual._
import is.hail.variant._

import scala.reflect.classTag

object CallFunctions extends RegistryFunctions {
  def registerAll() {
    registerWrappedScalaFunction1("Call", TString, TCall, (rt: Type, st: SType) => SCanonicalCall)(
      Call.getClass,
      "parse",
    )

    registerSCode1("callFromRepr", TInt32, TCall, (rt: Type, _: SType) => SCanonicalCall) {
      case (er, cb, rt, repr, _) => SCanonicalCall.constructFromIntRepr(cb, repr.asInt.value)
    }

    registerSCode1("Call", TBoolean, TCall, (rt: Type, _: SType) => SCanonicalCall) {
      case (er, cb, rt, phased, _) =>
        SCanonicalCall.constructFromIntRepr(
          cb,
          Code.invokeScalaObject[Int](
            Call0.getClass,
            "apply",
            Array(classTag[Boolean].runtimeClass),
            Array(phased.asBoolean.value),
          ),
        )
    }

    registerSCode2(
      "Call",
      TInt32,
      TBoolean,
      TCall,
      (rt: Type, _: SType, _: SType) => SCanonicalCall,
    ) {
      case (er, cb, rt, a1, phased, _) =>
        SCanonicalCall.constructFromIntRepr(
          cb,
          Code.invokeScalaObject[Int](
            Call1.getClass,
            "apply",
            Array(classTag[Int].runtimeClass, classTag[Boolean].runtimeClass),
            Array(a1.asInt.value, phased.asBoolean.value),
          ),
        )
    }

    registerSCode3(
      "Call",
      TInt32,
      TInt32,
      TBoolean,
      TCall,
      (rt: Type, _: SType, _: SType, _: SType) => SCanonicalCall,
    ) {
      case (er, cb, rt, a1, a2, phased, _) =>
        SCanonicalCall.constructFromIntRepr(
          cb,
          Code.invokeScalaObject[Int](
            Call2.getClass,
            "apply",
            Array(
              classTag[Int].runtimeClass,
              classTag[Int].runtimeClass,
              classTag[Boolean].runtimeClass,
            ),
            Array(a1.asInt.value, a2.asInt.value, phased.asBoolean.value),
          ),
        )
    }

    registerSCode1(
      "UnphasedDiploidGtIndexCall",
      TInt32,
      TCall,
      (rt: Type, _: SType) => SCanonicalCall,
    ) {
      case (er, cb, rt, x, _) =>
        SCanonicalCall.constructFromIntRepr(
          cb,
          Code.invokeScalaObject[Int](
            Call2.getClass,
            "fromUnphasedDiploidGtIndex",
            Array(classTag[Int].runtimeClass),
            Array(x.asInt.value),
          ),
        )
    }

    registerWrappedScalaFunction2(
      "Call",
      TArray(TInt32),
      TBoolean,
      TCall,
      {
        case (rt: Type, _: SType, _: SType) => SCanonicalCall
      },
    )(CallN.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities)
      registerSCode1(q, TCall, TBoolean, (rt: Type, _: SType) => SBoolean) {
        case (er, cb, rt, call, _) =>
          primitive(cb.memoize(Code.invokeScalaObject[Boolean](
            Call.getClass,
            q,
            Array(classTag[Int].runtimeClass),
            Array(call.asCall.canonicalCall(cb)),
          )))
      }

    registerSCode1("ploidy", TCall, TInt32, (rt: Type, _: SType) => SInt32) {
      case (er, cb, rt, call, _) =>
        primitive(cb.memoize(Code.invokeScalaObject[Int](
          Call.getClass,
          "ploidy",
          Array(classTag[Int].runtimeClass),
          Array(call.asCall.canonicalCall(cb)),
        )))
    }

    registerSCode1("unphase", TCall, TCall, (rt: Type, a1: SType) => a1) {
      case (er, cb, rt, call, _) =>
        call.asCall.unphase(cb)
    }

    registerSCode2(
      "containsAllele",
      TCall,
      TInt32,
      TBoolean,
      (rt: Type, _: SType, _: SType) => SBoolean,
    ) {
      case (er, cb, rt, call, allele, _) =>
        primitive(call.asCall.containsAllele(cb, allele.asInt.value))
    }

    registerSCode1("nNonRefAlleles", TCall, TInt32, (rt: Type, _: SType) => SInt32) {
      case (er, cb, rt, call, _) =>
        primitive(cb.memoize(Code.invokeScalaObject[Int](
          Call.getClass,
          "nNonRefAlleles",
          Array(classTag[Int].runtimeClass),
          Array(call.asCall.canonicalCall(cb)),
        )))
    }

    registerSCode1("unphasedDiploidGtIndex", TCall, TInt32, (rt: Type, _: SType) => SInt32) {
      case (er, cb, rt, call, _) =>
        primitive(cb.memoize(Code.invokeScalaObject[Int](
          Call.getClass,
          "unphasedDiploidGtIndex",
          Array(classTag[Int].runtimeClass),
          Array(call.asCall.canonicalCall(cb)),
        )))
    }

    registerSCode2("index", TCall, TInt32, TInt32, (rt: Type, _: SType, _: SType) => SInt32) {
      case (er, cb, rt, call, idx, _) =>
        primitive(cb.memoize(Code.invokeScalaObject[Int](
          Call.getClass,
          "alleleByIndex",
          Array(classTag[Int].runtimeClass, classTag[Int].runtimeClass),
          Array(call.asCall.canonicalCall(cb), idx.asInt.value),
        )))
    }

    registerSCode2(
      "downcode",
      TCall,
      TInt32,
      TCall,
      (rt: Type, _: SType, _: SType) => SCanonicalCall,
    ) {
      case (er, cb, rt, call, downcodedAllele, _) =>
        SCanonicalCall.constructFromIntRepr(
          cb,
          Code.invokeScalaObject[Int](
            Call.getClass,
            "downcode",
            Array(classTag[Int].runtimeClass, classTag[Int].runtimeClass),
            Array(call.asCall.canonicalCall(cb), downcodedAllele.asInt.value),
          ),
        )
    }

    registerSCode2(
      "lgt_to_gt",
      TCall,
      TArray(TInt32),
      TCall,
      { case (rt: Type, sc: SCall, _: SType) => sc },
    ) {
      case (er, cb, rt, call, localAlleles, errorID) =>
        call.asCall.lgtToGT(cb, localAlleles.asIndexable, errorID)
    }

    registerWrappedScalaFunction2(
      "oneHotAlleles",
      TCall,
      TInt32,
      TArray(TInt32),
      {
        case (rt: Type, _: SType, _: SType) => SIndexablePointer(PCanonicalArray(PInt32(true)))
      },
    )(Call.getClass, "oneHotAlleles")
  }
}
