package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {
  def apply(ir: IR): IR = ir match {
    case NA(t) => NA(t)
    case ArrayRange(x, y, z) => StreamRange(x, y, z)
    case ArrayMap(a, n, b) => ArrayMap(apply(a), n, b)
    case ArrayFilter(a, n, b) => ArrayFilter(apply(a), n, b)
    case Let(n, v, b) => Let(n, v, apply(b))
    case ir => ToStream(ir)
  }
}
