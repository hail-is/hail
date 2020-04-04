package is.hail.expr.types.physical

import is.hail.asm4s._
import is.hail.expr.types.virtual.TCall
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
