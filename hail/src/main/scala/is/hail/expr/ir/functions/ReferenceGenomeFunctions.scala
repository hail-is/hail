package is.hail.expr.ir.functions

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.{PBoolean, PCanonicalString, PInt32, PLocus, PString, PType}
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome

class ReferenceGenomeFunctions(registry: IRFunctionRegistry) extends RegistryFunctions(registry) {
  def rgCode(mb: EmitMethodBuilder[_], rg: ReferenceGenome): Code[ReferenceGenome] = mb.getReferenceGenome(rg)

  def registerAll() {
    registerCode1t("isValidContig", LocusFunctions.tlocus("R"), TString, TBoolean, (_: Type, _: PType) => PBoolean()) {
      case (r, rt, tlocus, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb, tlocus.asInstanceOf[TLocus].rg).invoke[String, Boolean]("isValidContig", scontig)
    }

    registerCode2t("isValidLocus", LocusFunctions.tlocus("R"), TString, TInt32, TBoolean, (_: Type, _: PType, _: PType) => PBoolean()) {
      case (r, rt, typeArg: TLocus, (contigT, contig: Code[Long]), (posT, pos: Code[Int])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb, typeArg.rg).invoke[String, Int, Boolean]("isValidLocus", scontig, pos)
    }

    registerPCode4t("getReferenceSequenceFromValidLocus",
      Array(LocusFunctions.tlocus("R")),
      TString, TInt32, TInt32, TInt32, TString,
      (_: Type, _: PType, _: PType, _: PType, _: PType) => PCanonicalString()) {
      case (r, cb, Seq(typeParam: TLocus), rt: PString, contig, pos, before, after) =>
        val scontig = contig.asString.loadString()
        unwrapReturn(cb, r.region, rt,
          rgCode(cb.emb, typeParam.rg).invoke[String, Int, Int, Int, String]("getSequence",
            scontig,
            pos.asInt.intCode(cb),
            before.asInt.intCode(cb),
            after.asInt.intCode(cb)))
    }

    registerCode1t("contigLength", LocusFunctions.tlocus("R"), TString, TInt32, (_: Type, _: PType) => PInt32()) {
      case (r, rt, typeArg: TLocus, (contigT, contig: Code[Long])) =>
        val scontig = asm4s.coerce[String](wrapArg(r, contigT)(contig))
        rgCode(r.mb, typeArg.rg).invoke[String, Int]("contigLength", scontig)
    }

    registerIR("getReferenceSequence", Array(TString, TInt32, TInt32, TInt32), TString, typeParameters = Array(LocusFunctions.tlocus("R"))) {
      case (tl, Seq(contig, pos, before, after)) =>
        val getRef = registry.lookupUnseeded(
          name = "getReferenceSequenceFromValidLocus",
          returnType = TString,
          typeParameters = tl,
          arguments = Seq(TString, TInt32, TInt32, TInt32)).get
        val isValid = registry.lookupUnseeded(
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
