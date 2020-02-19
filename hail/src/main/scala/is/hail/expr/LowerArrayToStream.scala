package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TIterable, TStream}

object LowerArrayToStream {
  private def nonStreamIterable(node: IR) = node.typ.isInstanceOf[TIterable] && !node.typ.isInstanceOf[TStream]

  private def boundary(node: IR): IR = {
    var streamified = streamify(node)

    if (streamified.typ.isInstanceOf[TStream] && node.typ.isInstanceOf[TArray])
      streamified = ToArray(streamified)

    assert(streamified.typ == node.typ)
    streamified
  }

  private def toStream(node: IR): IR = {
    val r = streamify(node)
    if(nonStreamIterable(r))
      ToStream(r)
    else
      r
  }

  private def streamify(node: IR): IR = {
    node match {
      case ArrayFold(a, zero, zn, an, body) => ArrayFold(toStream(a), boundary(zero), zn, an, boundary(body))
      case ArrayFor(a, n, b) => ArrayFor(toStream(a), n, boundary(b))
      case ArrayFold2(a, accum, valueName, seq, result) => ArrayFold2(
        toStream(a),
        accum.map { case (name, value) => (name, boundary(value)) },
        valueName, seq.map(boundary), boundary(result))
      case RunAggScan(a, name, init, seq, res, sig) => RunAggScan(toStream(a), name, boundary(init), boundary(seq), boundary(res), sig)
      case MakeArray(args, t) => MakeStream(args.map(boundary), TStream(t.elementType, t.required))
      case ArrayRange(start, stop, step) => StreamRange(boundary(start), boundary(stop), boundary(step))
      case ArrayZip(childIRs, names, body, behavior) => ArrayZip(childIRs.map(toStream), names, boundary(body), behavior)
      case ArrayMap(a, n, b) => ArrayMap(toStream(a), n, boundary(b))
      case ArrayFilter(a, n, b) => ArrayFilter(toStream(a), n, boundary(b))
      case ArrayFlatMap(a, n, b) => ArrayFlatMap(toStream(a), n, toStream(b))
      case ArrayScan(a, zero, zn, an, body) => ArrayScan(toStream(a), boundary(zero), zn, an, boundary(body))
      case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
        ArrayLeftJoinDistinct(toStream(l), toStream(r), ln, rn, boundary(keyf), boundary(joinf))
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        CollectDistributedArray(toStream(contextsIR), boundary(globalsIR), contextsName, globalsName,  boundary(bodyIR))
      case ToDict(a) => ToDict(toStream(a))
      case ToSet(a) => ToSet(toStream(a))
      case ArraySort(a, leftName, rightName, compareIR) => ArraySort(toStream(a), leftName, rightName, boundary(compareIR))
      case GroupByKey(collection) => GroupByKey(toStream(collection))
      case _ =>
        val newChildren = node.children.map(child => boundary(child.asInstanceOf[IR]))
        if ((node.children, newChildren).zipped.forall(_ eq _))
          node
        else
          node.copy(newChildren)
    }
  }

  def apply(node: IR): IR = boundary(node)
}
