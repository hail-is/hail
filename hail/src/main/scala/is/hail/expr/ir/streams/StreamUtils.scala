package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode, IR, NDArrayMap, NDArrayMap2, Ref, RunAggScan, StagedArrayBuilder, StreamFilter, StreamFlatMap, StreamFold, StreamFold2, StreamFor, StreamJoinRightDistinct, StreamMap, StreamScan, StreamZip, StreamZipJoin}
import is.hail.types.physical.{PCanonicalArray, PCode, PIndexableCode, SingleCodePCode}

trait StreamArgType {
  def apply(outerRegion: Region, eltRegion: Region): Iterator[java.lang.Long]
}

object StreamUtils {

  def toArray(
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    destRegion: Value[Region]
  ): PIndexableCode = {
    val mb = cb.emb

    val xLen = mb.newLocal[Int]("sta_len")
    val aTyp = PCanonicalArray(stream.element.st.canonicalPType(), true)
    stream.length match {
      case None =>
        val vab = new StagedArrayBuilder(stream.element.st.canonicalPType(), mb, 0)
        writeToArrayBuilder(cb, stream, vab, destRegion)
        cb.assign(xLen, vab.size)

        aTyp.constructFromElements(cb, destRegion, xLen, deepCopy = false) { (cb, i) =>
          IEmitCode(cb, vab.isMissing(i), PCode(aTyp.elementType, vab(i)))
        }

      case Some(len) =>

        var pushElem: (EmitCodeBuilder, IEmitCode) => Unit = null
        var finish: (EmitCodeBuilder) => PIndexableCode = null

        stream.memoryManagedConsume(destRegion, cb, setup = { cb =>
          cb.assign(xLen, len)
          val (_pushElem, _finish) = aTyp.constructFromFunctions(cb, destRegion, xLen, deepCopy = stream.requiresMemoryManagementPerElement)
          pushElem = _pushElem
          finish = _finish
        }) { cb =>
          pushElem(cb, stream.element.toI(cb))
        }

        finish(cb)
    }
  }

  def writeToArrayBuilder(
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    ab: StagedArrayBuilder,
    destRegion: Value[Region]
  ): Unit = {
    stream.memoryManagedConsume(destRegion, cb, setup = { cb =>
      cb += ab.clear
      cb += ab.ensureCapacity(stream.length.getOrElse(const(16)))

    }) { cb =>
      stream.element.toI(cb).consume(cb,
        cb += ab.addMissing(),
        sc => cb += ab.add(SingleCodePCode.fromPCode(cb, sc, destRegion, deepCopy = stream.requiresMemoryManagementPerElement).code)
      )
    }
  }

  private[ir] def multiplicity(root: IR, refName: String): Int = {
    var uses = 0

    // assumes no name collisions, a bit hacky...
    def traverse(ir: IR, mult: Int): Unit = ir match {
      case Ref(name, _) => if (refName == name) uses += mult
      case StreamMap(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamFilter(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamFlatMap(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamJoinRightDistinct(l, r, _, _, _, c, j, _) =>
        traverse(l, mult); traverse(r, mult); traverse(j, 2)
      case StreamScan(a, z, _, _, b) =>
        traverse(a, mult); traverse(z, 2); traverse(b, 2)
      case RunAggScan(a, _, i, s, r, _) =>
        traverse(a, mult); traverse(i, 2); traverse(s, 2); traverse(r, 2)
      case StreamZipJoin(as, _, _, _, f) =>
        as.foreach(traverse(_, mult)); traverse(f, 2)
      case StreamZip(as, _, body, _) =>
        as.foreach(traverse(_, mult)); traverse(body, 2)
      case StreamFold(a, zero, _, _, body) =>
        traverse(a, mult); traverse(zero, mult); traverse(body, 2)
      case StreamFold2(a, accs, _, seqs, res) =>
        traverse(a, mult)
        accs.foreach { case (_, acc) => traverse(acc, mult) }
        seqs.foreach(traverse(_, 2))
        traverse(res, 2)
      case StreamFor(a, _, body) =>
        traverse(a, mult); traverse(body, 2)
      case NDArrayMap(a, _, body) =>
        traverse(a, mult); traverse(body, 2)
      case NDArrayMap2(l, r, _, _, body) =>
        traverse(l, mult); traverse(r, mult); traverse(body, 2)

      case _ => ir.children.foreach {
        case child: IR => traverse(child, mult)
        case _ =>
      }
    }

    traverse(root, 1)
    uses min 2
  }

  def isIterationLinear(ir: IR, refName: String): Boolean =
    multiplicity(ir, refName) <= 1
}
