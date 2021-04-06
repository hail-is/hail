package is.hail.expr.ir

import is.hail.types.physical.StreamSingleCodeType
import is.hail.types.virtual.TStream


class StreamMemoryManagement(val m: Memo[StreamMemoType]) {
  def lookup(ir: IR): StreamMemoType = {
    assert(ir.typ.isInstanceOf[TStream])
    m.lookup(ir)
  }
}

case class StreamMemoType(separateRegions: Boolean, nested: Option[StreamMemoType]) {
  def nestedUnion(other: StreamMemoType): StreamMemoType = {
    StreamMemoType(separateRegions || other.separateRegions, (nested, other.nested) match {
      case (None, None) => None
      case (Some(n1), Some(n2)) => Some(n1.nestedUnion(n2))
    })
  }
}

object InferStreamMemoryManagement {
  def apply(ir: IR, usesAndDefs: UsesAndDefs): StreamMemoryManagement = {
    val m = Memo.empty[StreamMemoType]

    def lookup(name: String, defNode: IR): StreamMemoType = defNode match {
      case Let(`name`, value, _) =>
        m.lookup(value)
      case StreamMap(s, `name`, _) =>
        m.lookup(s).nested.get
      case x@StreamZip(as, _, _, _) =>
        m.lookup(as(x.nameIdx(name))).nested.get
      case StreamFilter(s, `name`, _) => m.lookup(s).nested.get
      case StreamFlatMap(s, `name`, _) => m.lookup(s).nested.get
      case StreamFor(s, `name`, _) => m.lookup(s).nested.get
      case StreamFold(s, _, _, `name`, _) => m.lookup(s).nested.get
      case StreamScan(s, _, _, `name`, _) => m.lookup(s).nested.get
      case StreamFold2(s, _, `name`, _, _) => m.lookup(s).nested.get
      case StreamJoinRightDistinct(left, _, _, _, `name`, _, _, _) => m.lookup(left).nested.get
      case StreamJoinRightDistinct(_, right, _, _, _, `name`, _, _) => m.lookup(right).nested.get
      case RunAggScan(s, `name`, _, _, _, _) => m.lookup(s).nested.get

    }

    def _inferBottomUp(x: IR): Unit = {
      x.children.foreach {
        case x: IR => _inferBottomUp(x)
      }

      if (x.typ.isInstanceOf[TStream]) {

        def getOpt(ir: IR): Option[StreamMemoType] = ir.typ match {
          case _: TStream => Some(m.lookup(ir))
          case _ => None
        }

        m.bind(x, x match {
          case ref: Ref => lookup(ref.name, usesAndDefs.defs.lookup(ref).asInstanceOf[IR])
          case Let(_, _, body) => m.lookup(body)
          case NA(t: TStream) =>
            var ts = t.elementType
            var nestingDepth = 0
            while (ts.isInstanceOf[TStream]) {
              ts = ts.asInstanceOf[TStream].elementType
              nestingDepth += 1
            }
            (0 until nestingDepth)
              .foldLeft(StreamMemoType(false, None)) { case (smt, _) => StreamMemoType(false, Some(smt)) }
          case ReadPartition(_, _, _) =>
            StreamMemoType(true, None)
          case In(_, SingleCodeEmitParamType(_, StreamSingleCodeType(separateRegions, _))) =>
            StreamMemoType(separateRegions, None)
          case MakeStream(_, _, separateRegions) =>
            StreamMemoType(separateRegions, None)
          case RunAggScan(s, _, _, _, _, _) =>
            m.lookup(s)
          case ShufflePartitionBounds(_, _) =>
            StreamMemoType(false, None)
          case ShuffleRead(_, _) =>
            StreamMemoType(true, None)
          case StreamJoinRightDistinct(left, right, _, _, _, _, join, _) =>
            StreamMemoType(m.lookup(left).separateRegions || m.lookup(right).separateRegions, getOpt(join))
          case StreamScan(s, _, _, _, _) =>
            StreamMemoType(m.lookup(s).separateRegions, None)
          case ToStream(_, separateRegions) =>
            StreamMemoType(separateRegions, None)
          case StreamTake(s, _) =>
            m.lookup(s)
          case StreamDrop(s, _) =>
            m.lookup(s)
          case StreamRange(_, _, _, separateRegions) =>
            StreamMemoType(separateRegions, None)
          case StreamGroupByKey(s, _) =>
            val parent = m.lookup(s)
            StreamMemoType(parent.separateRegions, Some(parent))
          case StreamGrouped(s, _) =>
            val parent = m.lookup(s)
            StreamMemoType(parent.separateRegions, Some(parent))
          case StreamFilter(s, _, _) =>
            m.lookup(s)
          case StreamMap(s, _, body) =>
            StreamMemoType(m.lookup(s).separateRegions, getOpt(body))
          case StreamFlatMap(s, _, body) =>
            val nested = m.lookup(body)
            StreamMemoType(m.lookup(s).separateRegions || nested.separateRegions, Some(nested))
          case StreamMerge(l, r, _) =>
            val ll = m.lookup(l)
            val rr = m.lookup(r)
            ll.nestedUnion(rr)
          case If(_, l, r) =>
            val ll = m.lookup(l)
            val rr = m.lookup(r)
            ll.nestedUnion(rr)
          case StreamMultiMerge(streams, _) =>
            val separateRegions = streams.map(m.lookup(_).separateRegions).reduce(_ || _)
            StreamMemoType(separateRegions, None)
          case StreamZip(streams, _, body, _) =>
            val separateRegions = streams.map(m.lookup(_).separateRegions).reduce(_ || _)
            StreamMemoType(separateRegions, getOpt(body))
          case StreamZipJoin(streams, _, _, _, joinF) =>
            val separateRegions = streams.map(m.lookup(_).separateRegions).reduce(_ || _)
            StreamMemoType(separateRegions, None)
        })
      }
    }

    _inferBottomUp(ir)

    new StreamMemoryManagement(m)
  }


}
