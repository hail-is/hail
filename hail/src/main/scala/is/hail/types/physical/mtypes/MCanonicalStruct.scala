package is.hail.types.physical.mtypes

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.BaseStruct
import is.hail.types.physical.stypes.{SInt32Value, SStructPointer, SType, SValue}
import is.hail.types.physical.{PBaseStruct, PType}
import is.hail.utils._

case class MField(typ: MType, name: String, idx: Int, required: Boolean)

object MCanonicalStruct {
  def getByteSizeAndOffsets(fields: IndexedSeq[MField], nMissingBytes: Long, byteOffsets: Array[Long]): Long = {
    assert(byteOffsets.length == fields.length)
    val bp = new BytePacker()

    var offset: Long = nMissingBytes
    fields.zipWithIndex.foreach { case (t, i) =>
      val fSize = t.typ.byteSize
      val fAlignment = t.typ.alignment

      bp.getSpace(fSize, fAlignment) match {
        case Some(start) =>
          byteOffsets(i) = start
        case None =>
          val mod = offset % fAlignment
          if (mod != 0) {
            val shift = fAlignment - mod
            bp.insertSpace(shift, offset)
            offset += (fAlignment - mod)
          }
          byteOffsets(i) = offset
          offset += fSize
      }
    }
    offset
  }

  def alignment(fields: IndexedSeq[MField]): Long = {
    if (fields.isEmpty)
      1
    else
      fields.map(_.typ.alignment).max
  }

}

class MCanonicalStruct(val fields: IndexedSeq[MField]) extends MStruct {

  val (missingIdx: Array[Int], nMissing: Int) = BaseStruct.getMissingIndexAndCount(fields.map(_.required).toArray)
  val nMissingBytes: Int = UnsafeUtils.packBitsToBytes(nMissing)
  val byteOffsets: Array[Long] = new Array[Long](size)
  override val byteSize: Long = MCanonicalStruct.getByteSizeAndOffsets(fields, nMissingBytes, byteOffsets)
  override val alignment: Long = MCanonicalStruct.alignment(fields)

  def size: Int = fields.size

  def getField(cb: EmitCodeBuilder, idx: Int, mv: MValue): IEmitMCode = {
    IEmitMCode(cb,
      isFieldMissing(mv, idx),
      getField(mv, idx)
    )
  }

  def storeFromSValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: SValue, region: Value[Region], deepCopy: Boolean): MValue = {
    value.typ match {
      case SStructPointer(t) if t == this =>
      // fast path
      case _ =>
      // store fields one at a time
    }
    memory.toMValue
  }

  def isFieldDefined(mv: MValue, fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(mv, fieldIdx)

  def isFieldMissing(mv: MValue, fieldIdx: Int): Code[Boolean] =
    if (fields(fieldIdx).required)
      false
    else
      Region.loadBit(mv.addr, missingIdx(fieldIdx).toLong)

  def getField(mv: MValue, fieldIdx: Int): MCode = {
    new MCode(mv.addr.get + byteOffsets(fieldIdx), fields(fieldIdx).typ)
  }

  override def pointerType: SStructPointer = SStructPointer(this)
}
