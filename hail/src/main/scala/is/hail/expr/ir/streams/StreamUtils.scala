package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode, StagedArrayBuilder}
import is.hail.types.physical.stypes.interfaces.SIndexableCode
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

        var pushElem: (EmitCodeBuilder, IEmitCode) => Unit = null
        var finish: (EmitCodeBuilder) => PIndexableCode = null

        stream.memoryManagedConsume(destRegion, cb, setup = { cb =>
          cb.assign(xLen, len)
          val (_pushElem, _finish) = aTyp.constructFromFunctions(cb, destRegion, xLen, deepCopy = stream.separateRegions)
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
        sc => cb += ab.add(SingleCodePCode.fromPCode(cb, sc, destRegion, deepCopy = stream.separateRegions).code)
      )
    }
  }
}
