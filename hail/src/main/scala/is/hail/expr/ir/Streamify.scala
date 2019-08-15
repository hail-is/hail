package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {

  /* Notes on the invariants of these functions:
   * - 'streamify' converts an IR with a TIterable type into an IR with type TStream.
   * - 'unstreamify' converts an IR with type TStream into an IR with some non-TStream
   *      TIterable type (such as TArray).
   * - 'apply' preserves an IR's type.
   */

  private[this] def streamify(streamableNode: IR): IR = streamableNode match {
    case ArrayRange(start, stop, step) => StreamRange(start, stop, step)
    case MakeArray(args, t) => MakeStream(args.map(apply), TStream(t.elementType, t.required))
    case ArrayMap(a, x, b) => ArrayMap(streamify(a), x, apply(b))
    case ArrayFilter(a, x, b) => ArrayFilter(streamify(a), x, apply(b))
    case ArrayFlatMap(a, x, b) => ArrayFlatMap(streamify(a), x, streamify(b))
    case ArrayScan(a, z, zn, vn, b) => ArrayScan(streamify(a), apply(z), zn, vn, apply(b))
    case ArraySort(a, l, r, c) => ToStream(ArraySort(streamify(a), l, r, apply(c)))
    case ArrayLeftJoinDistinct(l, r, ln, rn, k, j) =>
      ToStream( // TODO: remove this 'ToStream' when we implement streaming joins
        ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, apply(k), apply(j)))
    case ToArray(s) => streamify(s)
    case ToDict(s) => ToStream(ToDict(streamify(s)))
    case ToSet(s) => ToStream(ToSet(streamify(s)))
    case Let(n, v, b) => Let(n, apply(v), streamify(b))
    case _ =>
      if (streamableNode.typ.isInstanceOf[TStream])
        MapIR(apply)(streamableNode)
      else
        ToStream(MapIR(apply)(streamableNode))
  }

  private[this] def unstreamify(streamNode: IR): IR = {
    assert(streamNode.typ.isInstanceOf[TStream])
    streamNode match {
      case ToStream(c) => c
      case Let(n, v, b) => Let(n, v, unstreamify(b))
      case s => ToArray(s)
    }
  }

  def apply(node: IR): IR = node match {
    case ArrayFold(a, zero, zn, an, body) => ArrayFold(streamify(a), zero, zn, an, body)
    case ArrayFor(a, n, b) => ArrayFor(streamify(a), n, b)
    case _ =>
      if (node.typ.isInstanceOf[TIterable])
        unstreamify(streamify(node))
      else
        MapIR(apply)(node)
  }
}
