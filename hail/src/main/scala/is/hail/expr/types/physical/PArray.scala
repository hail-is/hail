package is.hail.expr.types.physical

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
import is.hail.utils._
import scala.reflect.{ClassTag, _}

abstract class PArray extends PContainer with PStreamable {
  def toConcrete(t: PArray) = t match {
    case x: PCanonicalArray => x
  }
}