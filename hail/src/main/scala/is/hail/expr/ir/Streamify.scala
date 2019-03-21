package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {

  private[this] def streamify(streamableNode: IR): IR = streamableNode match {
    case _: MakeStream | _: StreamRange | _: ReadPartition => streamableNode
    case ArrayRange(start, stop, step) => StreamRange(start, stop, step)
    case MakeArray(args, t) => MakeStream(args, TStream(t.elementType, t.required))
    case ArrayMap(a, n, b) =>
      if (a.typ.isInstanceOf[TStream]) streamableNode
      else ArrayMap(streamify(a), n, b)
    case ArrayFilter(a, n, b) =>
      if (a.typ.isInstanceOf[TStream]) streamableNode
      else ArrayFilter(streamify(a), n, b)
    case ArrayFlatMap(a, n, b) =>
      if (a.typ.isInstanceOf[TStream] && b.typ.isInstanceOf[TStream]) streamableNode
      else ArrayMap(streamify(a), n, streamify(b))
    case ArrayScan(a, zero, zn, an, body) =>
      if (a.typ.isInstanceOf[TStream]) streamableNode
      else ArrayScan(streamify(a), zero, zn, an, body)
    case ToArray(a) =>
      a.typ match {
        case _: TStream => a
        case _: TArray => streamify(a)
        case _ => ToStream(streamableNode)
      }
    case ToStream(a) =>
      a.typ match {
        case _: TStream => a
        case _ => streamableNode
      }
    case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
      ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, keyf, joinf)
    case _ =>
      ToStream(streamableNode)
  }

  private[this] def unstreamify(streamableNode: IR): IR = {
    val streamified = streamify(streamableNode)
    streamified match {
      case StreamRange(start, stop, step) => ArrayRange(start, stop, step)
      case MakeStream(args, t) => MakeArray(args, TArray(t.elementType, t.required))
      case ToStream(a) => a
      case _ => ToArray(streamableNode)
    }
  }

  def apply(node: IR): IR = node match {
    case ArraySort(a, l, r, comp) => ArraySort(streamify(a), l, r, comp)
    case ToSet(a) => ToSet(streamify(a))
    case ToDict(a) => ToDict(streamify(a))
    case ArrayFold(a, zero, zn, an, body) => ArrayFold(streamify(a), zero, zn, an, body)
    case ArrayFor(a, n, b) => ArrayFor(streamify(a), n, b)
    case _ if node.typ.isInstanceOf[TStreamable] => unstreamify(node)
    case _ => Copy(node, Children(node).map { case c: IR => apply(c) })
  }
}
