package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TStream}

object LowerArrayToStream {
  private def boundary(node: IR): IR = {
    var streamified = streamify(node)

    if (streamified.typ.isInstanceOf[TStream])
      streamified = ToArray(streamified)
    println(s"Streamified:\npre: ${node}:\npost:${streamified} \ntyp: ${streamified.typ} \nnode typ: ${node.typ}")
    assert(streamified.typ == node.typ)
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
      case MakeArray(args, t) => MakeStream(args.map(boundary), TStream(t.elementType, t.required))
      case ArrayRange(start, stop, step) => StreamRange(boundary(start), boundary(stop), boundary(step))
      case ArrayZip(childIRs, names, body, behavior) => ArrayZip(childIRs.map(streamify), names, boundary(body), behavior)
      case ArrayMap(a, n, b) => ArrayMap(streamify(a), n, boundary(b))
      case ArrayFilter(a, n, b) => ArrayFilter(streamify(a), n, boundary(b))
      case ArrayFlatMap(a, n, b) => ArrayFlatMap(streamify(a), n, streamify(b))
      case ArrayScan(a, zero, zn, an, body) => ArrayScan(streamify(a), boundary(zero), zn, an, boundary(body))
      case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
        ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, boundary(keyf), boundary(joinf))
      case x: ApplyIR => streamify(x.explicitNode)
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        ToStream(CollectDistributedArray(streamify(contextsIR), boundary(globalsIR), contextsName, globalsName,  boundary(bodyIR)))
      case Let(name, value, body) => Let(name, boundary(value), streamify(body))
      case ToArray(a) =>
        a.typ match {
          case _: TStream => node
          case _ => ToArray(streamify(a))
        }
      case _ =>
        val newChildren = node.children.map(child => boundary(child.asInstanceOf[IR]))
        val x = if ((node.children, newChildren).zipped.forall(_ eq _))
          node
        else
          node.copy(newChildren)

        if(x.typ.isInstanceOf[TArray]) {
          println(s"WRAPPING ${x}")
          ToStream(x)
        } else {
          x
        }
    }
  }

  def apply(node: IR): IR = {
    println(s"\n\nStarting LowerArrayToStream with: \n${ node }\n")
    val r = streamify(node)
    println(s"result of LowerArrayToStream is: ${r}\n\n")
    r
  }
}