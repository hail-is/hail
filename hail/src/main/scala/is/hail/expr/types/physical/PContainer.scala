package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.cxx
import is.hail.utils._

object PContainer {
  def loadLength(region: Region, aoff: Long): Int =
    region.loadInt(aoff)

  def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    region.loadInt(aoff)
}

abstract class PContainer extends PType {
  def elementType: PType

  def elementByteSize: Long

  override def byteSize: Long = 8

  def contentsAlignment: Long

  override def children = FastSeq(elementType)

  final def loadLength(region: Region, aoff: Long): Int =
    PContainer.loadLength(region, aoff)

  final def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    PContainer.loadLength(region, aoff)

  def _elementsOffset(length: Int): Long =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(4, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(4 + ((length + 7) >>> 3), elementType.alignment)

  def _elementsOffset(length: Code[Int]): Code[Long] =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(4, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(((length.toL + 7L) >>> 3) + 4L, elementType.alignment)

  var elementsOffsetTable: Array[Long] = _

  def elementsOffset(length: Int): Long = {
    if (elementsOffsetTable == null)
      elementsOffsetTable = Array.tabulate[Long](10)(i => _elementsOffset(i))

    if (length < 10)
      elementsOffsetTable(length)
    else
      _elementsOffset(length)
  }

  def elementsOffset(length: Code[Int]): Code[Long] = {
    // FIXME: incorporate table, maybe?
    _elementsOffset(length)
  }

  def contentsByteSize(length: Int): Long =
    elementsOffset(length) + length * elementByteSize

  def contentsByteSize(length: Code[Int]): Code[Long] = {
    elementsOffset(length) + length.toL * elementByteSize
  }

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean =
    !isElementDefined(region, aoff, i)

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean =
    elementType.required || !region.loadBit(aoff + 4, i)

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    !isElementDefined(region, aoff, i)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    if (elementType.required)
      true
    else
      !region.loadBit(aoff + 4, i.toL)

  def setElementMissing(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    region.setBit(aoff + 4, i)
  }

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] = {
    region.setBit(aoff + 4L, i.toL)
  }

  def setElementPresent(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    region.clearBit(aoff + 4, i)
  }

  def setElementPresent(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] = {
    region.clearBit(aoff + 4L, i.toL)
  }

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    aoff + elementsOffset(length) + i * elementByteSize

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long =
    elementOffset(aoff, loadLength(region, aoff), i)

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    aoff + elementsOffset(length) + i.toL * const(elementByteSize)

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    elementOffset(aoff, loadLength(region, aoff), i)

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Region, aoff: Long, i: Int): Long =
    loadElement(region, aoff, region.loadInt(aoff), i)

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    loadElement(region, aoff, region.loadInt(aoff), i)

  def allocate(region: Region, length: Int): Long = {
    region.allocate(contentsAlignment, contentsByteSize(length))
  }

  // FIXME expose intrinsic to just memset this
  private def writeMissingness(region: Region, aoff: Long, length: Int, value: Byte) {
    val nMissingBytes = (length + 7) / 8
    var i = 0
    while (i < nMissingBytes) {
      region.storeByte(aoff + 4 + i, value)
      i += 1
    }
  }

  def setAllMissingBits(region: Region, aoff: Long, length: Int) {
    if (elementType.required)
      return
    writeMissingness(region, aoff, length, -1)
  }

  def clearMissingBits(region: Region, aoff: Long, length: Int) {
    if (elementType.required)
      return
    writeMissingness(region, aoff, length, 0)
  }

  def initialize(region: Region, aoff: Long, length: Int, setMissing: Boolean = false) {
    region.storeInt(aoff, length)
    if (setMissing)
      setAllMissingBits(region, aoff, length)
    else
      clearMissingBits(region, aoff, length)
  }

  def initialize(region: Code[Region], aoff: Code[Long], length: Code[Int], a: Settable[Int]): Code[Unit] = {
    var c = region.storeInt(aoff, length)
    if (elementType.required)
      return c
    Code(
      c,
      a.store((length + 7) >>> 3),
      Code.whileLoop(a > 0,
        Code(
          a.store(a - 1),
          region.storeByte(aoff + 4L + a.toL, const(0))
        )
      )
    )
  }

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering =
    unsafeOrdering(this, missingGreatest)

  override def unsafeOrdering(rightType: PType, missingGreatest: Boolean): UnsafeOrdering = {
    require(this.isOfType(rightType))

    val right = rightType.asInstanceOf[PContainer]
    val eltOrd = elementType.unsafeOrdering(
      right.elementType,
      missingGreatest)

    new UnsafeOrdering {
      override def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        val length1 = loadLength(r1, o1)
        val length2 = right.loadLength(r2, o2)

        var i = 0
        while (i < math.min(length1, length2)) {
          val leftDefined = isElementDefined(r1, o1, i)
          val rightDefined = right.isElementDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val eOff1 = loadElement(r1, o1, length1, i)
            val eOff2 = right.loadElement(r2, o2, length2, i)
            val c = eltOrd.compare(r1, eOff1, r2, eOff2)
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
        Integer.compare(length1, length2)
      }
    }
  }

  def cxxLoadLength(a: cxx.Code): cxx.Code = {
    s"load_int($a)"
  }

  def cxxIsElementMissing(a: cxx.Code, i: cxx.Code): cxx.Code = {
    s"load_bit($a + 4, i)"
  }

  def cxxElementsOffset(len: cxx.Code): cxx.Code = {
    if (elementType.required)
      s"round_up_alignment(4, ${ elementType.alignment })"
    else
      s"round_up_alignment(4 + (($len + 7) >> 3), ${ elementType.alignment })"
  }

  def cxxElementOffset(a: cxx.Code, i: cxx.Code): cxx.Code = {
    s"$a + ${ cxxElementsOffset(cxxLoadLength(a)) } + i * ${ elementType.byteSize }"
  }

  def cxxLoadElement(a: cxx.EmitTriplet, i: cxx.EmitTriplet): cxx.EmitTriplet = {
    val __a = cxx.Variable("a", "char *")
    val __i = cxx.Variable("i", "int")
    val m = cxx.Variable("m", "bool")
    cxx.EmitTriplet(elementType, s"""
${ __a.define }
${ __i.define }
${ a.setup }
${ i.setup }
if (${ a.m } || ${ i.m })
  $m = true;
else {
  ${ __a } = $a.v;
  ${ __i } = $i.v;
  $m = ${ cxxIsElementMissing(__a.toString, __i.toString) }
}
""", m.toString,
      cxx.loadIRIntermediate(elementType, cxxElementOffset(__a.toString, __i.toString)))
  }
}
