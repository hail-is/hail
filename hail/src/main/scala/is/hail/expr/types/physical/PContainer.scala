package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.PType.storeShallow
import is.hail.utils._

object PContainer {
  def loadLength(aoff: Long): Int =
    Region.loadInt(aoff)

  def loadLength(aoff: Code[Long]): Code[Int] =
    Region.loadInt(aoff)

  def storeLength(aoff: Long, length: Int): Unit =
    Region.storeInt(aoff, length)

  def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit] =
    Region.storeInt(aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Long] = (len.toL + 7L) >>> 3

  def nMissingBytes(len: Int): Long = (len + 7L) >>> 3
}

abstract class PContainer extends PIterable {

  def elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  override def byteSize: Long = 8

  def contentsAlignment: Long = UnsafeUtils.nativeWordSize

  final def loadLength(region: Region, aoff: Long): Int =
    PContainer.loadLength(aoff)

  final def loadLength(aoff: Long): Int =
    PContainer.loadLength(aoff)

  final def loadLength(aoff: Code[Long]): Code[Int] =
    PContainer.loadLength(aoff)

  final def loadLength(region: Code[Region], aoff: Code[Long]): Code[Int] =
    loadLength(aoff)

  final def storeLength(aoff: Code[Long], length: Code[Int]): Code[Unit] =
    PContainer.storeLength(aoff, length)

  final def storeLength(region: Code[Region], aoff: Code[Long], length: Code[Int]): Code[Unit] =
    storeLength(aoff, length)

  def nMissingBytes(len: Code[Int]): Code[Long] = {
    if (elementType.required)
      const(0L)
    else
      PContainer.nMissingBytes(len)
  }

  def nMissingBytes(len: Int): Long = {
    if (elementType.required)
      0L
    else
      PContainer.nMissingBytes(len)
  }

  def lengthHeaderBytes: Long = 4

  def afterLengthHeaderAddress(aoff: Code[Long]) = {
    aoff + const(lengthHeaderBytes)
  }

  private def _headerOffset(length: Int): Long =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, UnsafeUtils.nativeWordSize)
    else
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes + PContainer.nMissingBytes(length), UnsafeUtils.nativeWordSize)

  private def _headerOffset(length: Code[Int]): Code[Long] =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, UnsafeUtils.nativeWordSize)
    else
      UnsafeUtils.roundUpAlignment(PContainer.nMissingBytes(length) + lengthHeaderBytes, UnsafeUtils.nativeWordSize)

  private lazy val lengthOffsetTable = 10
  private lazy val elementsOffsetTable: Array[Long] = Array.tabulate[Long](lengthOffsetTable)(i => _headerOffset(i))

  def elementsOffset(length: Int): Long = {
    if (length < lengthOffsetTable)
      elementsOffsetTable(length)
    else
      _headerOffset(length)
  }

  def elementsOffset(length: Code[Int]): Code[Long] =
    _headerOffset(length)

  def contentsByteSize(length: Int): Long =
    elementsOffset(length) + dataByteSize(length)

  def contentsByteSize(length: Code[Int]): Code[Long] =
    elementsOffset(length) + dataByteSize(length)

  def dataByteSize(length: Code[Int]): Code[Long] = {
    length.toL * elementByteSize
  }

  def dataByteSize(length: Int): Long = {
    length * elementByteSize
  }

  def isElementMissing(region: Region, aoff: Long, i: Int): Boolean =
    !isElementDefined(aoff, i)

  def isElementDefined(aoff: Long, i: Int): Boolean =
    elementType.required || !Region.loadBit(aoff + lengthHeaderBytes, i)

  def isElementDefined(region: Region, aoff: Long, i: Int): Boolean = isElementDefined(aoff, i)

  def isElementMissing(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    !isElementDefined(aoff, i)

  def isElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    isElementMissing(aoff, i)

  def isElementDefined(aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    if (elementType.required)
      true
    else
      !Region.loadBit(aoff + lengthHeaderBytes, i.toL)

  def isElementDefined(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Boolean] =
    isElementDefined(aoff, i)

  def setElementMissing(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    Region.setBit(aoff + lengthHeaderBytes, i)
  }

  def setElementMissing(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    Region.setBit(aoff + lengthHeaderBytes, i.toL)

  def setElementMissing(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] =
    setElementMissing(aoff, i)

  def setElementPresent(region: Region, aoff: Long, i: Int) {
    assert(!elementType.required)
    Region.clearBit(aoff + lengthHeaderBytes, i)
  }

  def setElementPresent(aoff: Code[Long], i: Code[Int]): Code[Unit] =
    Region.clearBit(aoff + lengthHeaderBytes, i.toL)

  def setElementPresent(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Unit] =
    setElementPresent(aoff, i)

  def firstElementAddress(aoff: Long, length: Int): Long =
    aoff + elementsOffset(length)

  def firstElementAddress(aoff: Code[Long], length: Code[Int]): Code[Long] =
    aoff + elementsOffset(length)

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    aoff + elementsOffset(length) + i * elementByteSize

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long =
    elementOffset(aoff, loadLength(region, aoff), i)

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    aoff + elementsOffset(length) + i.toL * const(elementByteSize)

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    elementOffset(aoff, loadLength(region, aoff), i)

  def nextElementAddress(currentOffset: Long) =
    currentOffset + elementByteSize

  def nextElementAddress(currentOffset: Code[Long]) =
    currentOffset + elementByteSize

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long = getElementAddress(aoff, length, i)

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def getElementAddress(aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def getElementAddress(aoff: Code[Long], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, Region.loadInt(aoff), i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Region, aoff: Long, i: Int): Long =
    getElementAddress(aoff, Region.loadInt(aoff), i)

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    getElementAddress(aoff, i)

  def read(region: Region, aoff: Long, i: Int): Any =
    UnsafeRow.read(elementType, region, loadElement(region, aoff, i))

  def read(region: Region, aoff: Long): Any =
    UnsafeRow.read(this, region, aoff)

  def allocate(region: Region, length: Int): Long =
    region.allocate(contentsAlignment, contentsByteSize(length))

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    region.allocate(contentsAlignment, contentsByteSize(length))

  private def writeMissingness(aoff: Long, length: Int, value: Byte) {
    Region.setMemory(aoff + lengthHeaderBytes, PContainer.nMissingBytes(length), value)
  }

  private def setAllMissingBits(aoff: Long, length: Int) {
    if (elementType.required)
      return

    writeMissingness(aoff, length, -1)
  }

  private def clearMissingBits(aoff: Long, length: Int) {
    if (elementType.required)
      return
    writeMissingness(aoff, length, 0)
  }

  def initialize(region: Region, aoff: Long, length: Int, setMissing: Boolean = false) {
    PContainer.storeLength(aoff, length)
    if (setMissing)
      setAllMissingBits(aoff, length)
    else
      clearMissingBits(aoff, length)
  }

  def stagedInitialize(aoff: Code[Long], length: Code[Int], setMissing: Boolean = false): Code[Unit] = {
    if (elementType.required)
      PContainer.storeLength(aoff, length)
    else
      Code(
        PContainer.storeLength(aoff, length),
        Region.setMemory(aoff + const(lengthHeaderBytes), nMissingBytes(length), const(if (setMissing) (-1).toByte else 0.toByte)))
  }

  override def unsafeOrdering(): UnsafeOrdering =
    unsafeOrdering(this)

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = {
    require(this.isOfType(rightType))

    val right = rightType.asInstanceOf[PContainer]
    val eltOrd = elementType.unsafeOrdering(
      right.elementType)

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
            return c
          }
          i += 1
        }
        Integer.compare(length1, length2)
      }
    }
  }

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], value: Code[Long], otherPT: PType, msg: String): Code[Long] = {
    val otherPTA = otherPT.asInstanceOf[PArray]
    assert(otherPTA.elementType.isPrimitive)
    val oldOffset = value
    val len = otherPTA.loadLength(oldOffset)
    if (otherPTA.elementType.required == elementType.required) {
      value
    } else {
      val newOffset = mb.newField[Long]
      Code(
        newOffset := allocate(r, len),
        stagedInitialize(newOffset, len),
        if (otherPTA.elementType.required) {
          // convert from required to non-required
          Code._empty
        } else {
          //  convert from non-required to required
          val i = mb.newField[Int]
          Code(
            i := 0,
            Code.whileLoop(i < len,
              otherPTA.isElementMissing(oldOffset, i).orEmpty(Code._fatal(s"${msg}: convertFrom $otherPT failed: element missing.")),
              i := i + 1
            )
          )
        },
        Region.copyFrom(otherPTA.elementOffset(oldOffset, len, 0), elementOffset(newOffset, len, 0), len.toL * elementByteSize),
        newOffset
      )
    }
  }

  def copyDataOfDifferentType(fb: FunctionBuilder[_], region: Code[Region], sourceType: PContainer, sourceValue: Code[Long], forceShallow: Boolean = false): Code[Long] = {
    val startOffset: ClassFieldRef[Long] = fb.newField[Long]
    val arraySize = fb.newField[Long]
    val elementLength = fb.newField[Int]
    val currentElementAddress = fb.newField[Long]
    val currentIdx= fb.newField[Int]

    var c = Code(
      elementLength := sourceType.loadLength(sourceValue),
      // TODO: do the right thing for PBaseStruct, PBinary
      arraySize := sourceType.contentsByteSize(elementLength),
      startOffset := region.allocate(this.contentsAlignment, arraySize)
    )

    if (this == sourceType) {
      Code(
        c,
        Region.copyFrom(sourceValue, startOffset, arraySize),
        startOffset
      )
    } else {
      // TODO: add cases for PBaseStruct, PBinary
      c = Code(c, PContainer.storeLength(startOffset, elementLength))
      c = Code(c, this.stagedInitialize(startOffset, elementLength))
      c = Code(c, currentElementAddress := this.firstElementAddress(startOffset, elementLength))
      c = Code(c, currentIdx.store(0))

      if (this.elementType == sourceType.elementType) {
        Code(
          c,
          Region.copyFrom(sourceType.afterLengthHeaderAddress(sourceValue), currentElementAddress, sourceType.dataByteSize(elementLength)),
          startOffset
        )
      } else {
        Code(c,
          currentIdx.store(0),
          Code.whileLoop(currentIdx < elementLength,
            // similarly, switch for different types
            if (sourceType.elementType.isPrimitive) {
              storeShallow(
                sourceType.getElementAddress(sourceValue, currentIdx),
                sourceType.elementType.fundamentalType,
                currentElementAddress
              )
            } else {
//              if(sourceType.elementType.required) {
                Region.storeAddress(
                  currentElementAddress,
                  this.elementType.asInstanceOf[PContainer].copyDataOfDifferentType(
                    fb,
                    region,
                    sourceType.elementType.asInstanceOf[PContainer],
                    sourceType.getElementAddress(sourceValue, currentIdx),
                    forceShallow
                  )
                )
//              }

            },
            // leaky abstraction
            currentElementAddress := this.nextElementAddress(currentElementAddress),
            currentIdx := currentIdx + const(1)
          ),
          startOffset
        )
      }
    }
  }

  override def containsPointers: Boolean = true
}
