package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {
  def apply(ir: IR): IR = ir match {
    case NA(t) => NA(t)
    case ArrayRange(x, y, z) => StreamRange(x, y, z)
    case Let(n, v, b) => Let(n, v, apply(b))
    case ir => ToStream(ir)
  }
}
