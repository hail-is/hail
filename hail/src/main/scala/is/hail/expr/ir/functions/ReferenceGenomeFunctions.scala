package is.hail.expr.ir.functions

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.concrete.SStringPointer
import is.hail.types.physical.stypes.primitives.{SBoolean, SInt32}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.{PBoolean, PCanonicalString, PInt32, PLocus, PString, PType}
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome

object ReferenceGenomeFunctions extends RegistryFunctions {
  def rgCode(mb: EmitMethodBuilder[_], rg: ReferenceGenome): Code[ReferenceGenome] = mb.getReferenceGenome(rg)

  def registerAll() {
    registerSCode1t("isValidContig", Array(LocusFunctions.tlocus("R")), TString, TBoolean, (_: Type, _: SType) => SBoolean) {
      case (r, cb, Seq(tlocus: TLocus), _, contig) =>
        val scontig = contig.asString.loadString()
        primitive(rgCode(r.mb, tlocus.asInstanceOf[TLocus].rg).invoke[String, Boolean]("isValidContig", scontig))
    }

    registerSCode2t("isValidLocus", Array(LocusFunctions.tlocus("R")), TString, TInt32, TBoolean, (_: Type, _: SType, _: SType) => SBoolean) {
      case (r, cb, Seq(tlocus: TLocus), _, contig, pos) =>
        val scontig = contig.asString.loadString()
        primitive(rgCode(r.mb, tlocus.rg).invoke[String, Int, Boolean]("isValidLocus", scontig, pos.asInt.intCode(cb)))
    }

    registerSCode4t("getReferenceSequenceFromValidLocus",
      Array(LocusFunctions.tlocus("R")),
      TString, TInt32, TInt32, TInt32, TString,
      (_: Type, _: SType, _: SType, _: SType, _: SType) => SStringPointer(PCanonicalString())) {
      case (r, cb, Seq(typeParam: TLocus), st, contig, pos, before, after) =>
        val scontig = contig.asString.loadString()
        unwrapReturn(cb, r.region, st,
          rgCode(cb.emb, typeParam.rg).invoke[String, Int, Int, Int, String]("getSequence",
            scontig,
            pos.asInt.intCode(cb),
            before.asInt.intCode(cb),
            after.asInt.intCode(cb)))
    }

    registerSCode1t("contigLength", Array(LocusFunctions.tlocus("R")), TString, TInt32, (_: Type, _: SType) => SInt32) {
      case (r, cb, Seq(tlocus: TLocus), _, contig) =>
        val scontig = contig.asString.loadString()
        primitive(rgCode(r.mb, tlocus.rg).invoke[String, Int]("contigLength", scontig))
    }

    registerIR("getReferenceSequence", Array(TString, TInt32, TInt32, TInt32), TString, typeParameters = Array(LocusFunctions.tlocus("R"))) {
      case (tl, Seq(contig, pos, before, after)) =>
        val getRef = IRFunctionRegistry.lookupUnseeded(
          name = "getReferenceSequenceFromValidLocus",
          returnType = TString,
          typeParameters = tl,
          arguments = Seq(TString, TInt32, TInt32, TInt32)).get
        val isValid = IRFunctionRegistry.lookupUnseeded(
          "isValidLocus",
          TBoolean,
          typeParameters = tl,
          Seq(TString, TInt32)).get

        val r = isValid(tl, Seq(contig, pos))
        val p = getRef(tl, Seq(contig, pos, before, after))
        If(r, p, NA(TString))
    }
  }
}
