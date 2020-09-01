package is.hail.types.physical

import is.hail.asm4s._
import is.hail.types.virtual.TCall
import is.hail.expr.ir.EmitCodeBuilder

abstract class PCall extends PType {
  lazy val virtualType: TCall.type = TCall
}

abstract class PCallValue extends PValue {
  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def forEachAllele(cb: EmitCodeBuilder)(alleleCode: Value[Int] => Unit): Unit
}

abstract class PCallCode extends PCode {
  def pt: PCall

  def ploidy(): Code[Int]

  def isPhased(): Code[Boolean]

  def memoize(cb: EmitCodeBuilder, name: String): PCallValue

  def memoizeField(cb: EmitCodeBuilder, name: String): PCallValue
}
