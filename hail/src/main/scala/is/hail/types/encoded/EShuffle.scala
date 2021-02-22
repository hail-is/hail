package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.concrete.{SCanonicalShufflePointer, SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable}
import is.hail.types.virtual._
import is.hail.utils._

case object EShuffleOptional extends EShuffle(false)

case object EShuffleRequired extends EShuffle(true)

class EShuffle(override val required: Boolean) extends EType {
  def _buildEncoder(cb: EmitCodeBuilder, pv: PValue, out: Value[OutputBuffer]): Unit = {
    pv.st match {
      case SCanonicalShufflePointer(t) =>
        val v = pv.asInstanceOf[SCanonicalShufflePointerSettable]
        val len = cb.newLocal[Int]("len", v.loadLength())
        cb += out.writeInt(len)
        cb += out.writeBytes(t.bytesAddress(v.shuffle.a), len)
    }
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): PCode = {
    val shuffleType = decodedPType(t).asInstanceOf[PCanonicalShuffle]
    val bT = shuffleType.representation
    val len = cb.newLocal[Int]("len", in.readInt())
    val barray = cb.newLocal[Long]("barray", bT.allocate(region, len))
    cb += bT.storeLength(barray, len)
    cb += in.readBytes(region, bT.bytesAddress(barray), len)
    new SCanonicalShufflePointerCode(SCanonicalShufflePointer(shuffleType), bT.loadCheapPCode(cb, barray))
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    cb += in.skipBytes(in.readInt())
  }

  def _decodedSType(requestedType: Type): SType = requestedType match {
    case t: TShuffle => SCanonicalShufflePointer(PCanonicalShuffle(t, required))
  }

  def _asIdent = "Shuffle"

  def _toPretty = "EShuffle"

  def setRequired(newRequired: Boolean): EShuffle = EShuffle(newRequired)
}

object EShuffle {
  def apply(required: Boolean = false): EShuffle = if (required) EShuffleRequired else EShuffleOptional
}
