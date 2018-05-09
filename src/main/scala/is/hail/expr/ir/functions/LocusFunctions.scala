package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types
import is.hail.asm4s
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.variant.Locus

object LocusFunctions extends RegistryFunctions {

  def getLocus(mb: EmitMethodBuilder, locus: Code[Long], typeString: String): Code[Locus] = {
    val tlocus = types.coerce[TLocus](tv(typeString).t)
    asm4s.coerce[Locus](wrapArg(mb, tlocus)(locus))
  }

  def registerAll() {
    registerCode("contig", tv("T", _.isInstanceOf[TLocus]), TString()) {
      case (mb, locus: Code[Long]) =>
        val locusObject = getLocus(mb, locus, "T")
        unwrapReturn(mb, TString())(locusObject.invoke[String]("contig"))
    }
  }
}
