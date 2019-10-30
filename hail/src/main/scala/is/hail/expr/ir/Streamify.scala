package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {

  private def stream(ir: IR): IR = ir match {
    case ToArray(s) => s
    case _ =>
      if (ir.typ.isInstanceOf[TStream])
        ir
      else {
        assert(ir.typ.isInstanceOf[TContainer])
        ToStream(ir)
      }
  }

  def apply(ir: IR): IR = MapIR(apply)(ir) match {
    case ArrayRange(x, y, z) => ToArray(StreamRange(x, y, z))
    case MakeArray(xs, t) => ToArray(MakeStream(xs, TStream(t.elementType, required = t.required)))
    //case x@ReadPartition(_, _, _) => x
    case ToStream(a) => ToArray(stream(a))
    case ArrayAggScan(a, n, q) => ToArray(ArrayAggScan(stream(a), n, q))
    case ArrayFilter(a, n, b) => ToArray(ArrayFilter(stream(a), n, b))
    case ArrayFlatMap(a, n, b) => ToArray(ArrayFlatMap(stream(a), n, stream(b)))
    case ArrayLeftJoinDistinct(l, r, ln, rn, k, j) => ToArray(ArrayLeftJoinDistinct(stream(l), stream(r), ln, rn, k, j))
    case ArrayMap(a, n, b) => ToArray(ArrayMap(stream(a), n, b))
    case ArrayScan(a, z, an, vn, b) => ToArray(ArrayScan(stream(a), z, an, vn, b))
    case ArrayAgg(a, n, q) => ArrayAgg(stream(a), n, q)
    case ArrayFold(a, z, an, vn, b) => ArrayFold(stream(a), z, an, vn, b)
    case ArrayFold2(a, acc, vn, seq, r) => ArrayFold2(stream(a), acc, vn, seq, r)
    case ArrayFor(a, vn, b) => ArrayFor(stream(a), vn, b)
    case ArraySort(a, ln, rn, c) => ArraySort(stream(a), ln, rn, c)
    case CollectDistributedArray(a, g, cn, gn, b) => CollectDistributedArray(stream(a), g, cn, gn, b)
    case GroupByKey(a) => GroupByKey(stream(a))
    case ToArray(a) => ToArray(stream(a))
    case ToDict(a) => ToDict(stream(a))
    case ToSet(a) => ToSet(stream(a))

    case a: ApplyIR =>
      val b = ApplyIR(a.function, a.args)
      b.conversion = args => apply(a.conversion(args))
      b.inline = a.inline
      b

    case x => x
  }
}
