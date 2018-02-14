package is.hail.expr.types

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.check.Gen
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, classTag}

abstract class TStructBase extends Type {
  def types: IndexedSeq[Type]
  
  def fieldRequired: IndexedSeq[Boolean] = types.map(_.required)

  override def children: Seq[Type] = types

  def size: Int = types.length

  def _toString: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _typeCheck(a: Any): Boolean =
    a.isInstanceOf[Row] && {
      val r = a.asInstanceOf[Row]
      r.length == types.length &&
        r.toSeq.zip(types).forall {
          case (v, t) => t.typeCheck(v)
        }
    }

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] = {
    if (types.isEmpty) {
      Gen.const(Annotation.empty)
    } else
      Gen.size.flatMap(fuel =>
        if (types.length > fuel)
          Gen.uniformSequence(types.map(t => if (t.required) t.genValue else Gen.const(null))).map(a => Annotation(a: _*))
        else
          Gen.uniformSequence(types.map(t => t.genValue)).map(a => Annotation(a: _*)))
  }

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && types.zip(a1.asInstanceOf[Row].toSeq).zip(a2.asInstanceOf[Row].toSeq)
      .forall {
        case ((t, x1), x2) =>
          t.valuesSimilar(x1, x2, tolerance)
      })

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val fieldOrderings = types.map(_.unsafeOrdering(missingGreatest)).toArray

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        while (i < types.length) {
          val leftDefined = isFieldDefined(r1, o1, i)
          val rightDefined = isFieldDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, loadField(r1, o1, i), r2, loadField(r2, o2, i))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            if (missingGreatest)
              return c
            else
              return -c
          }
          i += 1
        }
        0
      }
    }
  }

  val (missingIdx, nMissing) = {
    var i = 0
    val a = new Array[Int](types.length)
    types.zipWithIndex.foreach { case (t, idx) =>
      a(idx) = i
      if (!t.required)
        i += 1
    }
    (a, i)
  }

  def nMissingBytes: Int = (nMissing + 7) >>> 3

  var byteOffsets: Array[Long] = _
  override val byteSize: Long = {
    val a = new Array[Long](types.length)

    val bp = new BytePacker()

    var offset: Long = nMissingBytes
    types.zipWithIndex.foreach { case (t, i) =>
      val fSize = t.byteSize
      val fAlignment = t.alignment

      bp.getSpace(fSize, fAlignment) match {
        case Some(start) =>
          a(i) = start
        case None =>
          val mod = offset % fAlignment
          if (mod != 0) {
            val shift = fAlignment - mod
            bp.insertSpace(shift, offset)
            offset += (fAlignment - mod)
          }
          a(i) = offset
          offset += fSize
      }
    }
    byteOffsets = a
    offset
  }

  override val alignment: Long = {
    if (types.isEmpty)
      1
    else
      types.map(_.alignment).max
  }

  def allocate(region: Region): Long = {
    region.allocate(alignment, byteSize)
  }

  def clearMissingBits(region: Region, off: Long) {
    var i = 0
    while (i < nMissingBytes) {
      region.storeByte(off + i, 0)
      i += 1
    }
  }

  def clearMissingBits(region: Code[Region], off: Code[Long]): Code[Unit] = {
    var c: Code[Unit] = Code._empty
    var i = 0
    while (i < nMissingBytes) {
      c = Code(c, region.storeByte(off + i.toLong, const(0)))
      i += 1
    }
    c
  }

  def isFieldDefined(rv: RegionValue, fieldIdx: Int): Boolean =
    isFieldDefined(rv.region, rv.offset, fieldIdx)

  def isFieldDefined(region: Region, offset: Long, fieldIdx: Int): Boolean =
    fieldRequired(fieldIdx) || !region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    if (fieldRequired(fieldIdx))
      false
    else
      region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldDefined(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(region, offset, fieldIdx)

  def setFieldMissing(region: Region, offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    region.setBit(offset, missingIdx(fieldIdx))
  }

  def fieldOffset(offset: Long, fieldIdx: Int): Long =
    offset + byteOffsets(fieldIdx)

  def fieldOffset(offset: Code[Long], fieldIdx: Int): Code[Long] =
    offset + byteOffsets(fieldIdx)

  def loadField(rv: RegionValue, fieldIdx: Int): Long = loadField(rv.region, rv.offset, fieldIdx)

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    types(fieldIdx).fundamentalType match {
      case _: TArray | _: TBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadField(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Long] =
    loadField(region, fieldOffset(offset, fieldIdx), types(fieldIdx))

  private def loadField(region: Code[Region], fieldOffset: Code[Long], fieldType: Type): Code[Long] = {
    fieldType.fundamentalType match {
      case _: TArray | _: TBinary => region.loadAddress(fieldOffset)
      case _ => fieldOffset
    }
  }
}
