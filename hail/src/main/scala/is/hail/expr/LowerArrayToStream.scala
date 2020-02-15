package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TContainer, TIterable, TStream}

object LowerArrayToStream {
  private def nonStreamIterable(node: IR) = node.typ.isInstanceOf[TIterable] && !node.typ.isInstanceOf[TStream]

  private def boundary(node: IR): IR = {
    var streamified = streamify(node)

    if (streamified.typ.isInstanceOf[TStream] && nonStreamIterable(node)) {
      streamified = streamified match {
        case ToStream(a) => if (a.typ == node.typ) a else ToArray(a)
        case _ => ToArray(streamified)
      }
    } else if (nonStreamIterable(streamified) && node.typ.isInstanceOf[TStream])
      streamified = ToStream(streamified)

    assert(streamified.typ == node.typ)
    streamified
  }

  private def toStream(node: IR): IR = {
    if(nonStreamIterable(node)) {
      return ToStream(node)
    }
    // There are nodes that get matched on in both Emit an EmitStream contexts
    node
  }

  private def streamify(node: IR): IR = {
    node match {
      case ArraySort(a, l, r, comp) => ArraySort(toStream(streamify(a)), l, r, boundary(comp))
      case ArrayFold(a, zero, zn, an, body) => ArrayFold(toStream(streamify(a)), boundary(zero), zn, an, boundary(body))
      case ArrayFor(a, n, b) => ArrayFor(toStream(streamify(a)), n, boundary(b))
      case ArrayFold2(a, accum, valueName, seq, result) => ArrayFold2(
        toStream(streamify(a)),
        accum.map { case (name, value) => (name, boundary(value)) },
        valueName, seq.map(boundary), boundary(result))
      case RunAggScan(a, name, init, seq, res, sig) => RunAggScan(streamify(a), name, boundary(init), boundary(seq), boundary(res), sig)
      case MakeArray(args, t) => MakeStream(args.map(boundary), TStream(t.elementType, t.required))
      case ArrayRange(start, stop, step) => StreamRange(boundary(start), boundary(stop), boundary(step))
      case ArrayZip(childIRs, names, body, behavior) => ArrayZip(childIRs.map(streamify), names, boundary(body), behavior)
      case ArrayMap(a, n, b) => ArrayMap(toStream(streamify(a)), n, boundary(b))
      case ArrayFilter(a, n, b) => ArrayFilter(streamify(a), n, boundary(b))
      case ArrayFlatMap(a, n, b) => ArrayFlatMap(toStream(streamify(a)), n, toStream(streamify(b)))
      case ArrayScan(a, zero, zn, an, body) => ArrayScan(toStream(streamify(a)), boundary(zero), zn, an, boundary(body))
      case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
        ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, boundary(keyf), boundary(joinf))
      case x: ApplyIR => streamify(x.explicitNode)
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        CollectDistributedArray(toStream(streamify(contextsIR)), boundary(globalsIR), contextsName, globalsName,  boundary(bodyIR))
      case Let(name, value, body) => Let(name, boundary(value), streamify(body))
      case ToDict(a) => ToDict(toStream(streamify(a)))
      case ToSet(a) => ToSet(toStream(streamify(a)))
      case ArraySort(a, leftName, rightName, compareIR) => ArraySort(toStream(streamify(a)), leftName, rightName, boundary(compareIR))
      case GroupByKey(collection) => GroupByKey(toStream(streamify(collection)))
      case ToArray(a) =>
        a.typ match {
          case _: TArray => toStream(streamify(a))
          case _ => toStream(boundary(a))
        }
      case _ =>
        val newChildren = node.children.map(child => boundary(child.asInstanceOf[IR]))
        val x = if ((node.children, newChildren).zipped.forall(_ eq _))
          node
        else
          node.copy(newChildren)

        if(nonStreamIterable(x)) {
          println("HIT THE CONDTION")
          ToStream(x)
        }
        else
          x
    }
  }

  def apply(node: IR): IR = boundary(node)
}