package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TContainer, TStream, TStreamable}

object LowerArrayToStream {
  private def boundary(node: IR): IR = {
    var streamified = streamify(node)

    if (streamified.typ.isInstanceOf[TStream] && node.typ.isInstanceOf[TArray]) {
      println("Wrapping")
      streamified = ToArray(streamified)
    }

    if (streamified.typ.isInstanceOf[TArray] && node.typ.isInstanceOf[TStream])
      streamified = ToStream(streamified)

    assert(streamified.typ == node.typ)
    streamified
  }

  private def toStream(node: IR): IR = {
    assert(node.typ.isInstanceOf[TStreamable])
    node match {
      case _: ToStream => node
      case _ => {
        if(node.typ.isInstanceOf[TContainer]) {
          ToStream(node)
        } else {
          assert(node.typ.isInstanceOf[TStream])
          node
        }
      }
    }
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
      case ArrayFlatMap(a, n, b) => ArrayFlatMap(streamify(a), n, streamify(b))
      case ArrayScan(a, zero, zn, an, body) => ArrayScan(streamify(a), boundary(zero), zn, an, boundary(body))
      case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
        ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, boundary(keyf), boundary(joinf))
      case x: ApplyIR => streamify(x.explicitNode)
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        CollectDistributedArray(toStream(streamify(contextsIR)), boundary(globalsIR), contextsName, globalsName,  boundary(bodyIR))
      case Let(name, value, body) => Let(name, boundary(value), streamify(body))
      case ToDict(a) => ToDict(streamify(a))
      case ToSet(a) => ToSet(streamify(a))
      case ArraySort(a, leftName, rightName, compareIR) => ArraySort(streamify(a), leftName, rightName, boundary(compareIR))
      case ToArray(a) =>
        a.typ match {
          case _: TArray => ToStream(streamify(a))
          case _ => ToStream(boundary(a))
        }
      case _ =>
        val newChildren = node.children.map(child => boundary(child.asInstanceOf[IR]))
        val x = if ((node.children, newChildren).zipped.forall(_ eq _))
          node
        else
          node.copy(newChildren)

        if(x.typ.isInstanceOf[TArray]) {
          ToStream(x)
        } else {
          x
        }
    }
  }

  def apply(node: IR): IR = {
    val r = boundary(node)
    println(s"LowerArrayToStream: \npre: ${node}\npost: ${r}")
    r
  }
}