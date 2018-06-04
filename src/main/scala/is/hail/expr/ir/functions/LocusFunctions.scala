package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types
import is.hail.asm4s
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.variant.{Locus, RGBase, RGVariable, ReferenceGenome}
import is.hail.expr.ir._
import is.hail.utils.FastSeq

object LocusFunctions extends RegistryFunctions {

  def getLocus(mb: EmitMethodBuilder, locus: Code[Long], typeString: String): Code[Locus] = {
    val tlocus = types.coerce[TLocus](tv(typeString).t)
    Code.checkcast[Locus](wrapArg(mb, tlocus)(locus).asInstanceOf[Code[AnyRef]])
  }

  def registerLocusCode(methodName: String): Unit = {
    registerCode(methodName, tv("T", _.isInstanceOf[TLocus]), TBoolean()) {
      case (mb: EmitMethodBuilder, locus: Code[Long]) =>
        val locusObject = getLocus(mb, locus, "T")
        val tlocus = types.coerce[TLocus](tv("T").t)
        val rg = tlocus.rg.asInstanceOf[ReferenceGenome]

        val codeRG = mb.getReferenceGenome(rg)
        unwrapReturn(mb, TBoolean())(locusObject.invoke[RGBase, Boolean](methodName, codeRG))
    }
  }

  def registerAll() {
    registerCode("contig", tv("T", _.isInstanceOf[TLocus]), TString()) {
      case (mb, locus: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        val tlocus = types.coerce[TLocus](tv("T").t)
        tlocus.contig(region, locus)
    }

    registerCode("position", tv("T", _.isInstanceOf[TLocus]), TInt32()) {
      case (mb, locus: Code[Long]) =>
        val region = mb.getArg[Region](1).load()
        val tlocus = types.coerce[TLocus](tv("T").t)
        tlocus.position(region, locus)
    }

    registerLocusCode("isAutosomalOrPseudoAutosomal")
    registerLocusCode("isAutosomal")
    registerLocusCode("inYNonPar")
    registerLocusCode("inXPar")
    registerLocusCode("isMitochondrial")
    registerLocusCode("inXNonPar")
    registerLocusCode("inYPar")
  }
}
