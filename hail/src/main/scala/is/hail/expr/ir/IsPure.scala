package is.hail.expr.ir

import is.hail.types.virtual.TVoid

object IsPure {
  def apply(x: IR): Boolean = x match {
    case _ if x.typ == TVoid => false
    case _: WritePartition | _: WriteValue => false
    case _ => true
  }
}
