package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder, EmitMethodBuilder}
import is.hail.expr.types.virtual.TCall
import is.hail.variant.Genotype

object PCall {
  def apply(required: Boolean = false): PCall = PCanonicalCall(required)
}

abstract class PCall extends ComplexPType {
  lazy val virtualType: TCall.type = TCall

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    PInt32().codeOrdering(mb)
  }

  def ploidy(c: Code[Int]): Code[Int]

  def isPhased(c: Code[Int]): Code[Boolean]

  def forEachAllele(cb: EmitClassBuilder[_], _c: Code[Int], code: Code[Int] => Code[Unit]): Code[Unit]
}
