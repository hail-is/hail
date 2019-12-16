package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region, RegionValue, UnsafeOrdering, UnsafeUtils}
import is.hail.asm4s.{Code, const, _}
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.expr.types.BaseStruct
import is.hail.utils._

object PCanonicalBaseStruct {
  def getByteSizeAndOffsets(types: IndexedSeq[PType], nMissingBytes: Long, byteOffsets: Array[Long]): Long = {
    assert(byteOffsets.length == types.length)
    val bp = new BytePacker()

    var offset: Long = nMissingBytes
    types.zipWithIndex.foreach { case (t, i) =>
      val fSize = t.byteSize
      val fAlignment = t.alignment

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

  def alignment(types: Array[PType]): Long = {
    if (types.isEmpty)
      1
    else
      types.map(_.alignment).max
  }
}

trait PCanonicalBaseStruct extends PBaseStruct {
  lazy val missingIdx = new Array[Int](size)
  lazy val nMissing: Int = BaseStruct.getMissingness[PType](types, missingIdx)
  lazy val nMissingBytes = UnsafeUtils.packBitsToBytes(nMissing)
  lazy val byteOffsets = new Array[Long](size)
  override lazy val byteSize: Long = PCanonicalBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override lazy val alignment: Long = PCanonicalBaseStruct.alignment(types)

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  def identBase: String
  def _asIdent: String = {
    val sb = new StringBuilder
    sb.append(identBase)
    sb.append("_of_")
    types.foreachBetween { ty =>
      sb.append(ty.asIdent)
    } {
      sb.append("AND")
    }
    sb.append("END")
    sb.result()
  }

  def codeOrdering(mb: EmitMethodBuilder, so: Array[SortOrder]): CodeOrdering =
    codeOrdering(mb, this, so)

  def codeOrdering(mb: EmitMethodBuilder, other: PType, so: Array[SortOrder]): CodeOrdering

  def allocate(region: Region): Long =
    region.allocate(alignment, byteSize)

  def allocate(region: Code[Region]): Code[Long] = region.allocate(alignment, byteSize)

  override def containsPointers: Boolean = types.exists(_.containsPointers)
}
