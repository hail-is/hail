package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types
import is.hail.asm4s
import is.hail.variant.Locus

object LocusFunctions extends RegistryFunctions {

  def registerAll() {
    registerCode("contig", tv("T", _.isInstanceOf[TLocus]), TString()) {
      case (mb, locus: Code[Long]) =>
        val tlocus = types.coerce[TLocus](tv("T").t)
        val locusObject = asm4s.coerce[Locus](wrapArg(mb, tlocus)(locus))
        unwrapReturn(mb, TString())(locusObject.invoke[String]("contig"))
    }
  }
}
