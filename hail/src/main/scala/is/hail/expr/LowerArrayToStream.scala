package is.hail.expr.ir

import is.hail.expr.ir.lowering.StreamableIRs
import is.hail.expr.types.virtual.{TArray, TContainer, TIterable, TStream, TStreamable}

object LowerArrayToStream {

  private def boundary(node: IR): IR = {
    var streamified = streamify(node)

    if (streamified.typ.isInstanceOf[TStream] && node.typ.isInstanceOf[TContainer])
      streamified = ToArray(streamified)

    if (streamified.typ.isInstanceOf[TContainer] && node.typ.isInstanceOf[TStream])
      streamified = ToStream(streamified)
//    if (node.typ.isInstanceOf[TContainer] && !streamified.typ.isInstanceOf[TContainer]) {
//      streamified =  streamified match {
//        case ToStream(a) => a
//        case _ => {
//          assert(streamified.isInstanceOf[TIterable])
//          ToArray(streamified)
//        }
//      }
//    } else if (node.typ.isInstanceOf[TStream] && !streamified.typ.isInstanceOf[TStream]) {
//      streamified =  streamified match {
//        case ToArray(a) => toStream(a)
//        case _ => {
//          assert(streamified.isInstanceOf[TIterable])
//          toStream(streamified)
//        }
//      }
//    }

    if(streamified.typ != node.typ) {
      println(s"Fuck: \n\nstreamified:${streamified.typ}\n\nnode:${node.typ} for nodes \n\nstreamified:${streamified}\n\nnode: ${node}")
    }

    assert(streamified.typ == node.typ)
    streamified
  }

  private def toStream(node: IR, parentNode: IR): IR = {
    // We cannot make a stronger assertion here. There are cases when the node must be allowed through
    // even if it isn't TStreamable. For instance, Let nodes need to streamify some, but not all body
    // Because let is accepted in both Emit and EmitStream
    val x = node match {
      case ToStream(a) => a
      case ToArray(a) => a
      case _ => node
    }

    if(x.isInstanceOf[TIterable]) {
      ToStream(x)
    } else {
      x
    }
  }

  private def streamify(node: IR): IR = {
    node match {
      case ArraySort(a, l, r, comp) => ArraySort(toStream(streamify(a), node), l, r, boundary(comp))
      case ArrayFold(a, zero, zn, an, body) => ArrayFold(toStream(streamify(a), node), boundary(zero), zn, an, boundary(body))
      case ArrayFor(a, n, b) => ArrayFor(toStream(streamify(a), node), n, boundary(b))
      case ArrayFold2(a, accum, valueName, seq, result) => ArrayFold2(
        toStream(streamify(a), node),
        accum.map { case (name, value) => (name, boundary(value)) },
        valueName, seq.map(boundary), boundary(result))
      case RunAggScan(a, name, init, seq, res, sig) => RunAggScan(streamify(a), name, boundary(init), boundary(seq), boundary(res), sig)
      case MakeArray(args, t) => MakeStream(args.map(boundary), TStream(t.elementType, t.required))
      case ArrayRange(start, stop, step) => StreamRange(boundary(start), boundary(stop), boundary(step))
      case ArrayZip(childIRs, names, body, behavior) => ArrayZip(childIRs.map(streamify), names, boundary(body), behavior)
      case ArrayMap(a, n, b) => ArrayMap(toStream(streamify(a), node), n, boundary(b))
      case ArrayFilter(a, n, b) => ArrayFilter(streamify(a), n, boundary(b))
      case ArrayFlatMap(a, n, b) => ArrayFlatMap(toStream(streamify(a), node), n, toStream(streamify(b), node))
      case ArrayScan(a, zero, zn, an, body) => ArrayScan(toStream(streamify(a), node), boundary(zero), zn, an, boundary(body))
      case ArrayLeftJoinDistinct(l, r, ln, rn, keyf, joinf) =>
        ArrayLeftJoinDistinct(streamify(l), streamify(r), ln, rn, boundary(keyf), boundary(joinf))
      case x: ApplyIR => {
        println(s"IN APPLY IR FOR ${x}")
        streamify(x.explicitNode)
      }
      case CollectDistributedArray(contextsIR, globalsIR, contextsName, globalsName, bodyIR) =>
        CollectDistributedArray(toStream(streamify(contextsIR), node), boundary(globalsIR), contextsName, globalsName,  boundary(bodyIR))
      case Let(name, value, body) => {
        toStream(Let(name, boundary(value), boundary(streamify(body))), node)
      }
      case ToDict(a) => ToDict(toStream(streamify(a), node))
      case ToSet(a) => ToSet(toStream(streamify(a), node))
      case ArraySort(a, leftName, rightName, compareIR) => ArraySort(toStream(streamify(a), node), leftName, rightName, boundary(compareIR))
      case GroupByKey(collection) => GroupByKey(toStream(streamify(collection), node))
      case ToArray(a) =>
        a.typ match {
          case _: TArray => ToStream(streamify(a))
          case _ => ToStream(boundary(a))
        }
      case If(condIR, thenIR, elseIR) =>
        If(boundary(condIR), toStream(streamify(thenIR), node), toStream(streamify(elseIR), node))

      case _ =>
        val newChildren = node.children.map(child => boundary(child.asInstanceOf[IR]))
        node.copy(newChildren)
//      case _ =>
//        println(s"Node ${node} not allowed")
//        val newChildren = node.children.map(child => toStream(streamify(child.asInstanceOf[IR])))
//        toStream(node.copy(newChildren))
    }
  }

  def apply(node: IR): IR = {
    println(s"LowerArrayToStream: \npre: ${node}\n")
    val r = boundary(node)
    println(s"LowerArrayToStream: \npost: ${r}")
    r
  }
}