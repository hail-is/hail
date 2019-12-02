package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitMethodBuilder}
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

  def contentsAlignment: Long = elementType.alignment.max(4)

  override def byteSize: Long = 8

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

  private def _elementsOffset(length: Int): Long =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes + PContainer.nMissingBytes(length), elementType.alignment)

  private def _elementsOffset(length: Code[Int]): Code[Long] =
    if (elementType.required)
      UnsafeUtils.roundUpAlignment(lengthHeaderBytes, elementType.alignment)
    else
      UnsafeUtils.roundUpAlignment(PContainer.nMissingBytes(length) + lengthHeaderBytes, elementType.alignment)

  private lazy val lengthOffsetTable = 10
  private lazy val elementsOffsetTable: Array[Long] = Array.tabulate[Long](lengthOffsetTable)(i => _elementsOffset(i))

  def elementsOffset(length: Int): Long = {
    if (length < lengthOffsetTable)
      elementsOffsetTable(length)
    else
      _elementsOffset(length)
  }

  def elementsOffset(length: Code[Int]): Code[Long] =
    _elementsOffset(length)

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

<<<<<<< HEAD
  def firstElementAddress(aoff: Long, length: Int): Long =
    aoff + elementsOffset(length)

  def firstElementAddress(aoff: Code[Long], length: Code[Int]): Code[Long] =
    aoff + elementsOffset(length)

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    firstElementAddress(aoff, length) + i * elementByteSize
=======
  def firstElementOffset(aoff: Long, length: Int): Long =
    aoff + elementsOffset(length)

  def elementOffset(aoff: Long, length: Int, i: Int): Long =
    firstElementOffset(aoff, length) + i * elementByteSize
>>>>>>> bb68938df105fe5002203006d8c46ef9aebae0e0

  def elementOffsetInRegion(region: Region, aoff: Long, i: Int): Long =
    elementOffset(aoff, loadLength(region, aoff), i)

  def elementOffset(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
<<<<<<< HEAD
    firstElementAddress(aoff, length) + i.toL * const(elementByteSize)
=======
    firstElementOffset(aoff, length) + i.toL * const(elementByteSize)

  def firstElementOffset(aoff: Code[Long], length: Code[Int]): Code[Long] =
    aoff + elementsOffset(length)
>>>>>>> bb68938df105fe5002203006d8c46ef9aebae0e0

  def elementOffsetInRegion(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    elementOffset(aoff, loadLength(region, aoff), i)

  def nextElementAddress(currentOffset: Long) =
    currentOffset + elementByteSize

  def nextElementAddress(currentOffset: Code[Long]) =
    currentOffset + elementByteSize

  def loadElement(region: Region, aoff: Long, length: Int, i: Int): Long = loadElementAddress(aoff, length, i)

  def loadElement(region: Code[Region], aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] =
    loadElementAddress(aoff, length, i)

  def loadElementAddress(aoff: Long, length: Int, i: Int): Long = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElementAddress(aoff: Code[Long], i: Code[Int]): Code[Long] =
    loadElementAddress(aoff, PContainer.loadLength(aoff), i)

  def loadElementAddress(aoff: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = {
    val off = elementOffset(aoff, length, i)
    elementType.fundamentalType match {
      case _: PArray | _: PBinary => Region.loadAddress(off)
      case _ => off
    }
  }

  def loadElement(region: Region, aoff: Long, i: Int): Long =
    loadElementAddress(aoff, PContainer.loadLength(aoff), i)

  def loadElement(region: Code[Region], aoff: Code[Long], i: Code[Int]): Code[Long] =
    loadElementAddress(aoff, PContainer.loadLength(aoff), i)

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

  def ensureNoMissingValues(mb: EmitMethodBuilder, sourceOffset: Code[Long], sourceType: PContainer, onFail: Code[_]): Code[Unit] = {
    if(sourceType.elementType.required) {
      return Code._empty
    }

    val i = mb.newLocal[Long]
    Code(
<<<<<<< HEAD
      i := sourceType.nMissingBytes(sourceType.loadLength(sourceOffset)),
      Code.whileLoop(i > 0L,
        (i >= 8L).mux(
          Code(
            i := i - 8L,
            Region
              .loadLong(sourceOffset + sourceType.lengthHeaderBytes + i)
              .cne(const(0.toByte))
              .orEmpty(onFail)
          ),
          Code(
=======
      i := PContainer.nMissingBytes(sourceType.loadLength(sourceOffset)),
      Code.whileLoop(i > 0L,
        (i >= 8L).mux(
          Code(
            i := i - 8L,
            Region
              .loadLong(sourceOffset + sourceType.lengthHeaderBytes + i)
              .cne(const(0.toByte))
              .orEmpty(onFail)
          ),
          Code(
>>>>>>> bb68938df105fe5002203006d8c46ef9aebae0e0
            i := i - 1L,
            Region
              .loadByte(sourceOffset + sourceType.lengthHeaderBytes + i)
              .cne(const(0.toByte))
              .orEmpty(onFail)
          )
        )
      )
    )
  }

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], sourceOffset: Code[Long], sourceType: PContainer, msg: String): Code[Long] = {
    assert(sourceType.elementType.isPrimitive && this.isOfType(sourceType))

<<<<<<< HEAD
    if (sourceType.elementType.required == elementType.required) {
=======
    if (sourceType.elementType.required == this.elementType.required) {
>>>>>>> bb68938df105fe5002203006d8c46ef9aebae0e0
      return sourceOffset
    }

    val newOffset = mb.newField[Long]
    val len = sourceType.loadLength(sourceOffset)
    Code(
      ensureNoMissingValues(mb, sourceOffset, sourceType, Code._fatal(msg)),
      newOffset := allocate(r, len),
      stagedInitialize(newOffset, len),
      Region.copyFrom(sourceType.firstElementAddress(sourceOffset, len), firstElementAddress(newOffset, len), len.toL * elementByteSize),
      newOffset
    )
  }

  // TODO: non-shallow copy
  // TODO: handle PBaseStruct, PBinary
  def copyFromType(fb: EmitFunctionBuilder[_], region: Code[Region], sourcePType: PType, sourceOffset: Code[Long], forceShallow: Boolean = false): Code[Long] = {
    assert(this.isOfType(sourcePType))

    if (this == sourcePType) {

      return sourceOffset
    }

    val sourceType = sourcePType.asInstanceOf[PContainer]
    val destOffset: ClassFieldRef[Long] = fb.newField[Long]
    val arraySize = fb.newField[Long]
    val numberOfElements = fb.newField[Int]
    val currentElementAddress = fb.newField[Long]
    val currentIdx= fb.newField[Int]
    var c = Code(
      numberOfElements := sourceType.loadLength(sourceOffset),
      arraySize := sourceType.contentsByteSize(numberOfElements),
      destOffset := region.allocate(this.contentsAlignment, arraySize)
    )

    c = Code(c, PContainer.storeLength(destOffset, numberOfElements))
    c = Code(c, stagedInitialize(destOffset, numberOfElements))
    c = Code(c, currentElementAddress := this.firstElementAddress(destOffset, numberOfElements))

    if (this.elementType == sourceType.elementType) {
      return Code(
        c,
        Region.copyFrom(sourceType.afterLengthHeaderAddress(sourceOffset), currentElementAddress, sourceType.dataByteSize(numberOfElements)),
        destOffset
      )
    }

    var loopBody = if (sourceType.elementType.isPrimitive) {
      storeShallow(
        sourceType.loadElementAddress(sourceOffset, numberOfElements, currentIdx),
        sourceType.elementType.fundamentalType,
        currentElementAddress
      )
    } else {
      Region.storeAddress(
        currentElementAddress,
        this.elementType.asInstanceOf[PContainer].copyFromType(
          fb,
          region,
          sourceType.elementType.asInstanceOf[PContainer],
          sourceType.loadElementAddress(sourceOffset, numberOfElements, currentIdx),
          forceShallow
        )
      )
    }

    if(!this.elementType.required) {
      loopBody = sourceType.isElementMissing(sourceOffset, currentIdx).mux(this.setElementMissing(destOffset, currentIdx), loopBody)
    } else {
      c = Code(c, sourceType.ensureNoMissingValues(fb.apply_method, sourceOffset, sourceType, Code._fatal(
        "Found missing values. Cannot copy to type whose elements are required."
      )))
    }

    Code(
      c,
      currentIdx.store(0),
      Code.whileLoop(currentIdx < numberOfElements,
        loopBody,
        currentElementAddress := this.nextElementAddress(currentElementAddress),
        currentIdx := currentIdx + const(1)
      ),
      destOffset
    )
  }

  override def containsPointers: Boolean = true
}
