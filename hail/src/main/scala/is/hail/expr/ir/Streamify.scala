package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {
  def apply(ir: IR): IR = ir match {
    case NA(t: TStreamable) => NA(TStream(t.elementType, t.required))
    case MakeArray(xs, t) => MakeStream(xs, TStream(t.elementType, t.required))
    case ArrayRange(x, y, z) => StreamRange(x, y, z)
    case ArrayMap(a, n, b) => ArrayMap(apply(a), n, b)
    case ArrayZip(as, ns, b, behavior) => ArrayZip(as.map(apply), ns, b, behavior)
    case ArrayFilter(a, n, b) => ArrayFilter(apply(a), n, b)
    case ArrayFlatMap(a, n, b) => ArrayFlatMap(apply(a), n, apply(b))
    case ArrayLeftJoinDistinct(l, r, ln, rn, c, j) => ArrayLeftJoinDistinct(apply(l), apply(r), ln, rn, c, j)
    case ArrayScan(a, z, an, en, b) => ArrayScan(apply(a), z, an, en, b)
    case ArrayAggScan(a, n, q) => ArrayAggScan(apply(a), n, q)
    case RunAggScan(a, name, init, seq, res, sig) => RunAggScan(apply(a), name, init, seq, res, sig)
    case Let(n, v, b) => Let(n, v, apply(b))
    case If(c, t, e) => If(c, apply(t), apply(e))
    case x: ReadPartition => x
    case ir => ToStream(ir)
  }
}
