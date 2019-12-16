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
  lazy val fieldRequired: Array[Boolean] = types.map(_.required)

  lazy val fieldNames: Array[String] = fields.map(_.name).toArray

  lazy val missingIdx = new Array[Int](size)
  lazy val nMissing: Int = BaseStruct.getMissingness[PType](types, missingIdx)
  lazy val nMissingBytes = UnsafeUtils.packBitsToBytes(nMissing)
  lazy val byteOffsets = new Array[Long](size)
  override lazy val byteSize: Long = PCanonicalBaseStruct.getByteSizeAndOffsets(types, nMissingBytes, byteOffsets)
  override lazy val alignment: Long = PCanonicalBaseStruct.alignment(types)

  def index(str: String): Option[Int] = fieldIdx.get(str)

  def selfField(name: String): Option[PField] = fieldIdx.get(name).map(i => fields(i))

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

  override def unsafeOrdering(): UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = {
    require(this.isOfType(rightType))

    val right = rightType.asInstanceOf[PBaseStruct]
    val fieldOrderings: Array[UnsafeOrdering] =
      types.zip(right.types).map { case (l, r) => l.unsafeOrdering(r)}

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        while (i < types.length) {
          val leftDefined = isFieldDefined(r1, o1, i)
          val rightDefined = right.isFieldDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, loadField(r1, o1, i), r2, right.loadField(r2, o2, i))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            return c
          }
          i += 1
        }
        0
      }
    }
  }

  def allocate(region: Region): Long = {
    region.allocate(alignment, byteSize)
  }

  def allocate(region: Code[Region]): Code[Long] = region.allocate(alignment, byteSize)

  override def containsPointers: Boolean = types.exists(_.containsPointers)
}
