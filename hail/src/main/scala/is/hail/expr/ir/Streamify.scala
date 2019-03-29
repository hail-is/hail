package is.hail.expr.ir

import is.hail.expr.types.virtual._

object Streamify {

  private[this] def streamify(streamableNode: IR): IR = streamableNode match {
    case _: MakeStream | _: StreamRange | _: ReadPartition => Copy(streamableNode, Children(streamableNode).map { case c: IR => apply(c) } )
    case ArrayRange(start, stop, step) => StreamRange(apply(start), apply(stop), apply(step))
    case MakeArray(args, t) => MakeStream(args.map(apply), TStream(t.elementType, t.required))
    case ArrayMap(a, n, b) =>
      if (a.typ.isInstanceOf[TStream]) streamableNode
      else ArrayMap(streamify(a), n, apply(b))
    case ArrayFilter(a, n, b) =>
      if (a.typ.isInstanceOf[TStream]) streamableNode
      else ArrayFilter(streamify(a), n, apply(b))
    case ArrayFlatMap(a, n, b) =>
      if (a.typ.isInstanceOf[TStream] && b.typ.isInstanceOf[TStream]) streamableNode
      else ArrayFlatMap(streamify(a), n, streamify(b))
    case ArrayScan(a, zero, zn, an, body) =>
      if (a.typ.isInstanceOf[TStream]) streamableNode
      else ArrayScan(streamify(a), apply(zero), zn, an, apply(body))
    case ToArray(a) =>
      a.typ match {
        case _: TStream => a
        case _: TArray => streamify(a)
        case _ => ToStream(apply(streamableNode))
      }
    case ToStream(a) =>
      a.typ match {
        case _: TStream => a
        case _ => ToStream(apply(a))
      }
    case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
      ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, apply(keyf), apply(joinf))
    case Let(n, v, b) =>
      Let(n, apply(v), streamify(b))
    case _ =>
      ToStream(Copy(streamableNode, Children(streamableNode).map { case c: IR => apply(c) } ))
  }

  private[this] def unstreamify(streamableNode: IR): IR = streamableNode match {
    case ToArray(a) =>
      a.typ match {
        case _: TArray => ToArray(streamify(a))
        case _ => streamableNode
      }
    case ToStream(a) =>
      a.typ match {
        case _: TStream => ToArray(a)
        case _ => a
      }
    case If(cond, cnsq, altr) =>
      If(cond, unstreamify(cnsq), unstreamify(altr))
    case Let(n, v, b) =>
      Let(n, v, unstreamify(b))
    case _ =>
      streamify(streamableNode) match {
        case ToStream(a) if !a.typ.isInstanceOf[TStream] => a
        case s => ToArray(s)
      }
  }

  def apply(node: IR): IR = node match {
    case ArraySort(a, l, r, comp) => ArraySort(streamify(a), l, r, comp)
    case ToSet(a) => ToSet(streamify(a))
    case ToDict(a) => ToDict(streamify(a))
    case ArrayFold(a, zero, zn, an, body) => ArrayFold(streamify(a), zero, zn, an, body)
    case ArrayFor(a, n, b) => ArrayFor(streamify(a), n, b)
    case x: ApplyIR => apply(x.explicitNode)
    case _ if node.typ.isInstanceOf[TStreamable] => unstreamify(node)
    case _ => Copy(node, Children(node).map { case c: IR => apply(c) })
  }
}
