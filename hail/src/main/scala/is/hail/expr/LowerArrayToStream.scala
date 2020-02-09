package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TStream, TStreamable}

// The minimum set of nodes that need streamify (what emitArrayIterator is called on)
// things that call emitArrayIterator
// For instance, we will encounter MakeStruct, with a bunch of fields
// ArrayAgg
// CollectDistributedArray
// ArraySort
// ToSet
// ToDict
// ArrayMap | _: ArrayZip | _: ArrayFilter | _: ArrayRange | _: ArrayFlatMap | _: ArrayScan | _: ArrayLeftJoinDistinct | _: RunAggScan | _: ArrayAggScan | _: ReadPartition
// ArrayFold
// ArrayFold2
// ArrayFor
// ArrayAgg
// RunAgg
object LowerArrayToStream {
  private[this] def streamify(streamableNode: IR): IR = {
    println(s"\nstarting streamify with ${streamableNode}")

    val  r = streamableNode match {
      case _: MakeStream | _: StreamRange | _: ReadPartition => Copy(streamableNode, Children(streamableNode).map { case c: IR => apply(c) })
      case MakeArray(args, t) => MakeStream(args.map(apply), TStream(t.elementType, t.required))
      case ArrayRange(start, stop, step) => StreamRange(apply(start), apply(stop), apply(step))
      case ArrayZip(childIRs, names, body, behavior) => ArrayZip(childIRs.map(streamify), names, body, behavior)
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
      case _ =>
        ToStream(Copy(streamableNode, Children(streamableNode).map { case c: IR => apply(c) }))
    }

    println(s"ending streamify with ${r}\n")
    r
  }

  private[this] def unstreamify(streamableNode: IR): IR = {
    println(s"\nstarting unstreamify with ${streamableNode}")
    val r = streamableNode match {
      case ToArray(a) =>
        a.typ match {
          case _: TArray => ToArray(streamify(a))
          case _ => streamableNode
        }
      case ToStream(a) =>
        a.typ match {
          case _: TStream =>
            ToArray(a)
          case _ => a
        }
      case _ =>
        streamify(streamableNode) match {
          case ToStream(a) => a
          case s => ToArray(s)
        }
    }

    println(s"ending unstreamify with ${r}\n")
    r
  }

  def apply(node: IR): IR = {
    println(s"\n\nStarting LowerArrayToStream with: ${ node }")
    val r = node match {
      case ArraySort(a, l, r, comp) => ArraySort(streamify(a), l, r, comp)
      case ArrayFold(a, zero, zn, an, body) => ArrayFold(streamify(a), zero, zn, an, body)
      case ArrayFor(a, n, b) => ArrayFor(streamify(a), n, b)
      case ArrayFold2(a, accum, valueName, seq, result) => ArrayFold2(streamify(a), accum, valueName, seq, result)
      case x: ApplyIR => apply(x.explicitNode)
      case RunAggScan(a, name, init, seq, res, sig) => RunAggScan(streamify(a), name, init, seq, res, sig)
      case _ => unstreamify(Copy(node, Children(node).map { case c: IR => apply(c) }))
    }

    println(s"ending LowerArrayToStream with: ${ r }\n\n")
    r
  }
}