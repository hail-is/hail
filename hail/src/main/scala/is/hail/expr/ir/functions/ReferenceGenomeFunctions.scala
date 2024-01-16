package is.hail.expr.ir.functions
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.concrete.SJavaString
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBoolean, SInt32}
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome

object ReferenceGenomeFunctions extends RegistryFunctions {
  def rgCode(mb: EmitMethodBuilder[_], rg: String): Code[ReferenceGenome] =
    mb.getReferenceGenome(rg)

  def registerAll(): Unit = {
    registerSCode1t(
      "isValidContig",
      Array(LocusFunctions.tlocus("R")),
      TString,
      TBoolean,
      (_: Type, _: SType) => SBoolean,
    ) {
      case (r, cb, Seq(tlocus: TLocus), _, contig, _) =>
        val scontig = contig.asString.loadString(cb)
        primitive(cb.memoize(rgCode(r.mb, tlocus.asInstanceOf[TLocus].rg).invoke[String, Boolean](
          "isValidContig",
          scontig,
        )))
    }

    registerSCode2t(
      "isValidLocus",
      Array(LocusFunctions.tlocus("R")),
      TString,
      TInt32,
      TBoolean,
      (_: Type, _: SType, _: SType) => SBoolean,
    ) {
      case (r, cb, Seq(tlocus: TLocus), _, contig, pos, _) =>
        val scontig = contig.asString.loadString(cb)
        primitive(cb.memoize(rgCode(r.mb, tlocus.rg).invoke[String, Int, Boolean](
          "isValidLocus",
          scontig,
          pos.asInt.value,
        )))
    }

    registerSCode4t(
      "getReferenceSequenceFromValidLocus",
      Array(LocusFunctions.tlocus("R")),
      TString,
      TInt32,
      TInt32,
      TInt32,
      TString,
      (_: Type, _: SType, _: SType, _: SType, _: SType) => SJavaString,
    ) {
      case (r, cb, Seq(typeParam: TLocus), st, contig, pos, before, after, _) =>
        val scontig = contig.asString.loadString(cb)
        unwrapReturn(
          cb,
          r.region,
          st,
          rgCode(cb.emb, typeParam.rg).invoke[String, Int, Int, Int, String](
            "getSequence",
            scontig,
            pos.asInt.value,
            before.asInt.value,
            after.asInt.value,
          ),
        )
    }

    registerSCode1t(
      "contigLength",
      Array(LocusFunctions.tlocus("R")),
      TString,
      TInt32,
      (_: Type, _: SType) => SInt32,
    ) {
      case (r, cb, Seq(tlocus: TLocus), _, contig, _) =>
        val scontig = contig.asString.loadString(cb)
        primitive(cb.memoize(rgCode(r.mb, tlocus.rg).invoke[String, Int]("contigLength", scontig)))
    }

    registerIR(
      "getReferenceSequence",
      Array(TString, TInt32, TInt32, TInt32),
      TString,
      typeParameters = Array(LocusFunctions.tlocus("R")),
    ) {
      case (tl, Seq(contig, pos, before, after), _) =>
        val getRef = IRFunctionRegistry.lookupUnseeded(
          name = "getReferenceSequenceFromValidLocus",
          returnType = TString,
          typeParameters = tl,
          arguments = Seq(TString, TInt32, TInt32, TInt32),
        ).get
        val isValid = IRFunctionRegistry.lookupUnseeded(
          "isValidLocus",
          TBoolean,
          typeParameters = tl,
          Seq(TString, TInt32),
        ).get

        val r = isValid(tl, Seq(contig, pos), ErrorIDs.NO_ERROR)
        val p = getRef(tl, Seq(contig, pos, before, after), ErrorIDs.NO_ERROR)
        If(r, p, NA(TString))
    }
  }
}
