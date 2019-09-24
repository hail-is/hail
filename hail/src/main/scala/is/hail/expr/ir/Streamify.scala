package is.hail.expr.ir

import is.hail.expr.types.virtual._
import scala.annotation.tailrec

object Streamify {

  @tailrec
  private def expandFunctions(ir: IR): IR = ir match {
    case air: ApplyIR => expandFunctions(air.explicitNode)
    case _ => ir
  }

  def apply(ir0: IR): IR = MapIR(apply)(expandFunctions(ir0)) match {
    case ir@ReadPartition(path, spec, t) =>
      ToArray(ir)
    case ArrayRange(i, j, k) =>
      ToArray(StreamRange(i, j, k))
    case MakeArray(xs, t) =>
      ToArray(MakeStream(xs, TStream(t.elementType, t.required)))
    case ArrayMap(a, n, b) =>
      ToArray(ArrayMap(toStream(a), n, b))
    case ArrayFilter(a, n, c) =>
      ToArray(ArrayFilter(toStream(a), n, c))
    case ArrayFlatMap(a, n, c) =>
      ToArray(ArrayFlatMap(toStream(a), n, toStream(c)))
    case ArrayScan(a, z, an, vn, b) =>
      ToArray(ArrayScan(toStream(a), z, an, vn, b))
    case ArrayAggScan(a, n, q) =>
      ToArray(ArrayAggScan(toStream(a), n, q))
    case ArrayLeftJoinDistinct(l, r, ln, rn, key, join) =>
      ToArray(ArrayLeftJoinDistinct(toStream(l), toStream(r), ln, rn, key, join))
    case ArraySort(a, ln, rn, cmp) =>
      ArraySort(toStream(a), ln, rn, cmp)
    case ArrayFold(a, z, an, vn, b) =>
      ArrayFold(toStream(a), z, an, vn, b)
    case ArrayFold2(a, acc, vn, seq, res) =>
      ArrayFold2(toStream(a), acc, vn, seq, res)
    case ArrayFor(a, n, b) =>
      ArrayFor(toStream(a), n, b)
    case ArrayAgg(a, n, q) =>
      ArrayAgg(toStream(a), n, q)
    case CollectDistributedArray(ctxs, gs, cn, gn, b) =>
      CollectDistributedArray(toStream(ctxs), gs, cn, gn, b)
    case GroupByKey(a) => GroupByKey(toStream(a))
    case ToArray(a) => ToArray(toStream(a))
    case ToDict(a) => ToDict(toStream(a))
    case ToSet(a) => ToSet(toStream(a))
    case ir => ir
  }

  def toStream(ir: IR): IR = ir match {
    case If(c, thn, els) => If(c, toStream(thn), toStream(els))
    case Let(n, v, b) => Let(n, v, toStream(b))
    case ToArray(s) => assert(s.typ.isInstanceOf[TStream]); s
    case ToStream(a) => assert(a.typ.isInstanceOf[TContainer]); ir
    case a => assert(a.typ.isInstanceOf[TContainer]); ToStream(a)
  }
}
