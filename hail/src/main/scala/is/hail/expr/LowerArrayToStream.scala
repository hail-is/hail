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
  private def boundary(node: IR): IR = {
    val streamified = streamify(node)
    if (streamified.typ.isInstanceOf[TStream])
      ToArray(streamified)
    else
      streamified
  }

  private def streamify(node: IR): IR = {
    node match {
      case ArraySort(a, l, r, comp) => ArraySort(streamify(a), l, r, boundary(comp))
      case ArrayFold(a, zero, zn, an, body) => ArrayFold(streamify(a), boundary(zero), zn, an, boundary(body))
      case ArrayFor(a, n, b) => ArrayFor(streamify(a), n, boundary(b))
      case ArrayFold2(a, accum, valueName, seq, result) => ArrayFold2(
        streamify(a),
        accum.map { case (name, value) => (name, boundary(value)) },
        valueName, seq.map(boundary), boundary(result))
      case RunAggScan(a, name, init, seq, res, sig) => RunAggScan(streamify(a), name, boundary(init), boundary(seq), boundary(res), sig)
      case _: MakeStream | _: StreamRange | _: ReadPartition => node.copy(node.children.map(c => boundary(c.asInstanceOf[IR])))
      case MakeArray(args, t) => MakeStream(args.map(boundary), TStream(t.elementType, t.required))
      case ArrayRange(start, stop, step) => StreamRange(boundary(start), boundary(stop), boundary(step))
      case ArrayZip(childIRs, names, body, behavior) => ArrayZip(childIRs.map(streamify), names, boundary(body), behavior)
      case ArrayMap(a, n, b) => ArrayMap(streamify(a), n, boundary(b))
      case ArrayFilter(a, n, b) => ArrayFilter(streamify(a), n, boundary(b))
      case ArrayFlatMap(a, n, b) => ArrayFlatMap(streamify(a), n, streamify(b))
      case ArrayScan(a, zero, zn, an, body) => ArrayScan(streamify(a), boundary(zero), zn, an, boundary(body))
      case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
        ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, boundary(keyf), boundary(joinf))
      case x if x.typ.isInstanceOf[TArray] => ToStream(x)
      case _ =>
        val newChildren = node.children.map(child => boundary(child.asInstanceOf[IR]))
        if ((node.children, newChildren).zipped.forall(_ eq _))
          node
        else
          node.copy(newChildren)
    }
  }

  def apply(node: IR): IR = {
    println(s"\n\nStarting LowerArrayToStream with: ${ node }")
    streamify(node)
  }
}