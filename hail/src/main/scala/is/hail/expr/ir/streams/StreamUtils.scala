package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode, StagedArrayBuilder}
import is.hail.types.physical.{PCanonicalArray, PCode, PIndexableCode, SingleCodePCode}
import is.hail.utils._

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
        if (stream.separateRegions)
          cb.assign(stream.elementRegion, Region.stagedCreate(Region.REGULAR, destRegion.getPool()))
        else
          cb.assign(stream.elementRegion, destRegion)

        cb.assign(xLen, len)
        val (pushElem, finish) = aTyp.constructFromFunctions(cb, destRegion, xLen, deepCopy = stream.separateRegions)

        stream.consume(cb) { cb =>
          pushElem(cb, stream.element.toI(cb))
          if (stream.separateRegions)
            cb += stream.elementRegion.clearRegion()
        }

        if (stream.separateRegions)
          cb += stream.elementRegion.freeRegion()

        finish(cb)
    }
  }

  def writeToArrayBuilder(
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    ab: StagedArrayBuilder,
    destRegion: Value[Region]
  ): Unit = {
    if (stream.separateRegions)
      cb.assign(stream.elementRegion, Region.stagedCreate(Region.REGULAR, destRegion.getPool()))
    else
      cb.assign(stream.elementRegion, destRegion)

    cb += ab.clear
    cb += ab.ensureCapacity(stream.length.getOrElse(const(16)))
    stream.consume(cb) { cb =>
      stream.element.toI(cb).consume(cb,
        cb += ab.addMissing(),
        sc => cb += ab.add(SingleCodePCode.fromPCode(cb, sc, destRegion, deepCopy = stream.separateRegions).code)
      )

      if (stream.separateRegions)
        cb += stream.elementRegion.clearRegion()
    }

    if (stream.separateRegions)
      cb += stream.elementRegion.freeRegion()
  }
}
