package is.hail.types.physical

import is.hail.asm4s._
import is.hail.types.virtual.TCall
import is.hail.expr.ir.EmitCodeBuilder

abstract class PCall extends ComplexPType {
  lazy val virtualType: TCall.type = TCall
}

object PCallValue {
  def apply(pt: PCall, call: Settable[_]): PCallValue = pt match {
    case t: PCanonicalCall => new PCanonicalCallSettable(t, coerce[Int](call))
  }
}

abstract class PCallValue extends PValue {
  def ploidy()(implicit line: LineNumber): Code[Int]

  def isPhased()(implicit line: LineNumber): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit)(implicit line: LineNumber): Unit
}

abstract class PCallCode extends PCode {
  def pt: PCall

  def ploidy()(implicit line: LineNumber): Code[Int]

  def isPhased()(implicit line: LineNumber): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PCallValue
}
