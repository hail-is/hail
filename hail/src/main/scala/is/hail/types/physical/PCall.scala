package is.hail.types.physical

import is.hail.asm4s._
import is.hail.types.virtual.TCall
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SCallCode, SCallValue}

abstract class PCall extends PType {
  lazy val virtualType: TCall.type = TCall
}

abstract class PCallValue extends PValue with SCallValue {
  def ploidy()(implicit line: LineNumber): Code[Int]

  def isPhased()(implicit line: LineNumber): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit
}

abstract class PCallCode extends PCode with SCallCode {
  def pt: PCall

  def ploidy()(implicit line: LineNumber): Code[Int]

  def isPhased()(implicit line: LineNumber): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): PCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PCallValue
}
