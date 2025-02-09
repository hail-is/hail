package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s.Value
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{
  SCanonicalRNGStateValue, SRNGState, SRNGStateStaticInfo, SRNGStateStaticSizeValue,
}
import is.hail.types.virtual.Type
import is.hail.utils._

//case object ERNGStateOptional extends ERNGState(false)
//
//case object ERNGStateRequired extends ERNGState(true)

final case class ERNGState(override val required: Boolean, staticInfo: Option[SRNGStateStaticInfo])
    extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    (staticInfo, v) match {
      case (Some(staticInfo), v: SRNGStateStaticSizeValue) =>
        assert(staticInfo == v.staticInfo)
        for (x <- v.runningSum) cb += out.writeLong(x)
        for (x <- v.lastDynBlock) cb += out.writeLong(x)
      case (None, v: SCanonicalRNGStateValue) =>
        for (x <- v.runningSum) cb += out.writeLong(x)
        for (x <- v.lastDynBlock) cb += out.writeLong(x)
        cb += out.writeInt(v.numWordsInLastBlock)
        cb += out.writeBoolean(v.hasStaticSplit)
        cb += out.writeInt(v.numDynBlocks)
    }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = staticInfo match {
    case Some(staticInfo) =>
      new SRNGStateStaticSizeValue(
        _decodedSType(t),
        Array.fill(4)(cb.memoize(in.readLong())),
        Array.fill(staticInfo.numWordsInLastBlock)(cb.memoize(in.readLong())),
      )
    case None =>
      new SCanonicalRNGStateValue(
        _decodedSType(t),
        Array.fill(4)(cb.memoize(in.readLong())),
        Array.fill(4)(cb.memoize(in.readLong())),
        cb.memoize(in.readInt()),
        cb.memoize(in.readBoolean()),
        cb.memoize(in.readInt()),
      )
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    staticInfo match {
      case Some(staticInfo) =>
        for (_ <- 0 until (4 + staticInfo.numWordsInLastBlock))
          cb += in.skipLong()
      case None =>
        for (_ <- 0 until 8)
          cb += in.skipLong()
        cb += in.skipInt()
        cb += in.skipBoolean()
        cb += in.skipInt()
    }

  def _decodedSType(requestedType: Type): SRNGState = SRNGState(staticInfo)

  def _asIdent = "rngstate"

  def _toPretty = "ERNGState"

  def setRequired(newRequired: Boolean): ERNGState = ERNGState(newRequired, staticInfo)
}
