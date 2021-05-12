package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SIndexablePointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SInt32}
import is.hail.types.physical.{PCanonicalArray, PInt32}
import is.hail.types.virtual._
import is.hail.variant._

import scala.reflect.classTag

object CallFunctions extends RegistryFunctions {
  def registerAll() {
    registerWrappedScalaFunction1("Call", TString, TCall, (rt: Type, st: SType) => SCanonicalCall)(Call.getClass, "parse")

    registerPCode1("Call", TBoolean, TCall, (rt: Type, _: SType) => SCanonicalCall) {
      case (er, cb, rt, phased) =>
        
        SCanonicalCall.constructFromIntRepr(Code.invokeScalaObject[Int](
          Call0.getClass, "apply", Array(classTag[Boolean].runtimeClass), Array(phased.asBoolean.boolCode(cb))))
    }

    registerPCode2("Call", TInt32, TBoolean, TCall, (rt: Type, _: SType, _: SType) => SCanonicalCall) {
      case (er, cb, rt, a1, phased) =>
        SCanonicalCall.constructFromIntRepr(Code.invokeScalaObject[Int](
          Call1.getClass, "apply", Array(classTag[Int].runtimeClass, classTag[Boolean].runtimeClass), Array(a1.asInt.intCode(cb), phased.asBoolean.boolCode(cb))))
    }

    registerPCode3("Call", TInt32, TInt32, TBoolean, TCall, (rt: Type, _: SType, _: SType, _: SType) => SCanonicalCall) {
      case (er, cb, rt, a1, a2, phased) =>
        SCanonicalCall.constructFromIntRepr(Code.invokeScalaObject[Int](
          Call2.getClass, "apply", Array(classTag[Int].runtimeClass, classTag[Int].runtimeClass, classTag[Boolean].runtimeClass), Array(a1.asInt.intCode(cb), a2.asInt.intCode(cb), phased.asBoolean.boolCode(cb))))
    }

    registerPCode1("UnphasedDiploidGtIndexCall", TInt32, TCall, (rt: Type, _: SType) => SCanonicalCall) {
      case (er, cb, rt, x) =>
        SCanonicalCall.constructFromIntRepr(Code.invokeScalaObject[Int](
          Call2.getClass, "fromUnphasedDiploidGtIndex", Array(classTag[Int].runtimeClass), Array(x.asInt.intCode(cb))))
    }


    registerWrappedScalaFunction2("Call", TArray(TInt32), TBoolean, TCall, {
      case (rt: Type, _: SType, _: SType) => SCanonicalCall
    })(CallN.getClass, "apply")

    val qualities = Array("isPhased", "isHomRef", "isHet",
      "isHomVar", "isNonRef", "isHetNonRef", "isHetRef")
    for (q <- qualities) {
      registerPCode1(q, TCall, TBoolean, (rt: Type, _: SType) => SBoolean) {
        case (er, cb, rt, call) =>
          primitive(Code.invokeScalaObject[Boolean](
            Call.getClass, q, Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb))))
      }
    }

    registerPCode1("ploidy", TCall, TInt32, (rt: Type, _: SType) => SInt32) {
      case (er, cb, rt, call) =>
        primitive(Code.invokeScalaObject[Int](
          Call.getClass, "ploidy", Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb))))
    }

    registerPCode1("nNonRefAlleles", TCall, TInt32, (rt: Type, _: SType) => SInt32) {
      case (er, cb, rt, call) =>
        primitive(Code.invokeScalaObject[Int](
          Call.getClass, "nNonRefAlleles", Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb))))
    }

    registerPCode1("unphasedDiploidGtIndex", TCall, TInt32, (rt: Type, _: SType) => SInt32) {
      case (er, cb, rt, call) =>
        primitive(Code.invokeScalaObject[Int](
          Call.getClass, "unphasedDiploidGtIndex", Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb))))
    }

    registerPCode2("index", TCall, TInt32, TInt32, (rt: Type, _: SType, _: SType) => SInt32) {
      case (er, cb, rt, call, idx) =>
        primitive(Code.invokeScalaObject[Int](
          Call.getClass, "alleleByIndex", Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb), idx.asInt.intCode(cb))))
    }

    registerPCode2("downcode", TCall, TInt32, TCall, (rt: Type, _: SType, _: SType) => SCanonicalCall) {
      case (er, cb, rt, call, downcodedAllele) =>
        SCanonicalCall.constructFromIntRepr(Code.invokeScalaObject[Int](
          Call.getClass, "downcode", Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb), downcodedAllele.asInt.intCode(cb))))
    }

    registerPCode2("downcode", TCall, TInt32, TCall, (rt: Type, _: SType, _: SType) => SCanonicalCall) {
      case (er, cb, rt, call, downcodedAllele) =>
        SCanonicalCall.constructFromIntRepr(Code.invokeScalaObject[Int](
          Call.getClass, "downcode", Array(classTag[Int].runtimeClass), Array(call.asCall.loadCanonicalRepresentation(cb), downcodedAllele.asInt.intCode(cb))))
    }

    registerWrappedScalaFunction2("oneHotAlleles", TCall, TInt32, TArray(TInt32), {
      case (rt: Type, _: SType, _: SType) => SIndexablePointer(PCanonicalArray(PInt32(true)))
    })(Call.getClass, "oneHotAlleles")
  }
}
