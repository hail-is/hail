package is.hail.types.physical
import is.hail.annotations.{Annotation, Region, UnsafeOrdering, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.utils._

final case class PNewArray(elementType: PType, required: Boolean = false) extends PArray {
  // A C-struct for arrays with the following structure:
  // struct {
  //    int length;
  //    T *elements;
  //    char *missing; // if !required
  // }
  override val alignment: Long = 8
  val byteSize: Long = if (required) 16 else 24
  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)
  val contentsAlignment: Long = elementType.alignment
  private val elementsOffset: Long = 8
  private val missingOffset: Long = 16
  private def elements(addr: Long): Long = Region.loadLong(addr + elementsOffset)
  private def elements(addr: Code[Long]): Code[Long] = Region.loadAddress(addr + elementsOffset)
  private def missing(addr: Long): Long = Region.loadLong(addr + missingOffset)
  private def missing(addr: Code[Long]): Code[Long] = Region.loadAddress(addr + missingOffset)
  def nMissingBytes(len: Int): Int = UnsafeUtils.packBitsToBytes(len)
  def nMissingBytes(len: Code[Int]): Code[Int] = UnsafeUtils.packBitsToBytes(len)

  def elementIterator(addr: Long, length: Int): PArrayIterator = ???

  def loadLength(addr: Long): Int = Region.loadInt(addr)

  def loadLength(addr: Code[Long]): Code[Int] = Region.loadInt(addr)

  def storeLength(addr: Code[Long], length: Code[Int]): Code[Unit] = Region.storeInt(addr, length)

  def lengthHeaderBytes: Long = 0

  def elementsOffset(length: Int): Long = 0

  def elementsOffset(length: Code[Int]): Code[Long] = 0

  def isElementMissing(addr: Long, i: Int): Boolean = Region.loadBit(missing(addr), i)

  def isElementDefined(addr: Long, i: Int): Boolean = !isElementMissing(addr, i)

  def isElementMissing(addr: Code[Long], i: Code[Int]): Code[Boolean] = Region.loadBit(missing(addr), i.toL)

  def isElementDefined(addr: Code[Long], i: Code[Int]): Code[Boolean] = !isElementMissing(addr, i)

  def setElementMissing(addr: Long, i: Int): Unit =
    if (!elementRequired) {
      Region.setBit(missing(addr), i)
    }

  def setElementMissing(addr: Code[Long], i: Code[Int]): Code[Unit] =
    if (!elementRequired)
      Region.setBit(missing(addr), i.toL)
    else
      Code._fatal[Unit](s"Required element cannot be missing")

  def setElementPresent(addr: Long, i: Int) {
    if (!elementRequired)
      Region.clearBit(missing(addr), i)
  }

  def setElementPresent(addr: Code[Long], i: Code[Int]): Code[Unit] =
    if (!elementRequired)
      Region.clearBit(missing(addr), i.toL)
    else
      Code._empty

  def firstElementOffset(addr: Long, length: Int): Long = elements(addr)

  def firstElementOffset(addr: Code[Long], length: Code[Int]): Code[Long] = firstElementOffset(addr)

  def firstElementOffset(addr: Code[Long]): Code[Long] = elements(addr)

  def elementOffset(addr: Long, length: Int, i: Int): Long = elementOffset(addr, i)

  def elementOffset(addr: Long, i: Int): Long = elements(addr) + i * elementByteSize

  def elementOffset(addr: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = elementOffset(addr, i)

  def elementOffset(addr: Code[Long], i: Code[Int]): Code[Long] = firstElementOffset(addr) + i.toL * elementByteSize

  def loadElement(addr: Long, length: Int, i: Int): Long = loadElement(addr, i)

  def loadElement(addr: Long, i: Int): Long = elementType.unstagedLoadFromNested(elementOffset(addr, i))

  def loadElement(addr: Code[Long], i: Code[Int]): Code[Long] = elementType.loadFromNested(elementOffset(addr, i))

  def loadElement(addr: Code[Long], length: Code[Int], i: Code[Int]): Code[Long] = loadElement(addr, i)

  def allocate(region: Region, length: Int): Long = {
    val a = region.allocate(alignment, byteSize)
    val e = region.allocate(contentsAlignment, elementByteSize * length)
    Region.storeInt(a, length)
    Region.storeAddress(elements(a), e)
    if (!elementRequired) {
      val m = region.allocate(4, nMissingBytes(length))
      Region.storeAddress(missing(a), m)
    }
    a
  }

  def allocate(region: Code[Region], length: Code[Int]): Code[Long] =
    Code.memoize(region, "new_array_region", length, "new_array_length") { (r, len) =>
      Code.memoize(r.allocate(alignment, byteSize), "new_array_a") { a =>
        val e = r.allocate(contentsAlignment, elementByteSize * len)
        Code(
          Region.storeAddress(elements(a), e),
          if (elementRequired)
            Code._empty
          else {
            val m = r.allocate(4L, nMissingBytes(len))
            Region.storeAddress(missing(a), m)
          },
          a
        )
      }
    }

  private def writeMissingness(addr: Long, length: Int, value: Byte) {
    Region.setMemory(missing(addr), nMissingBytes(length), value)
  }

  def setAllMissingBits(addr: Long, length: Int): Unit =
    if (!elementRequired) {
      writeMissingness(addr, length, -1)
    }

  def clearMissingBits(addr: Long, length: Int): Unit =
    if (!elementRequired) {
      writeMissingness(addr, length, 0)
    }

  def initialize(addr: Long, length: Int, setMissing: Boolean): Unit = {
    storeLength(addr, length)
    if (setMissing)
      setAllMissingBits(addr, length)
    else
      clearMissingBits(addr, length)
  }

  def stagedInitialize(addr: Code[Long], length: Code[Int], setMissing: Boolean): Code[Unit] =
    if (elementRequired) {
      storeLength(addr, length)
    } else {
      Code.memoize(addr, "staged_init_addr",
        length, "staged_init_length") { (addr, length) =>
        Code(
          Region.storeInt(addr, length),
          Region.setMemory(missing(addr), nMissingBytes(length).toL, const(if (setMissing) (-1).toByte else 0.toByte)))

      }
    }

  def zeroes(region: Region, length: Int): Long = ???

  def zeroes(mb: EmitMethodBuilder[_], region: Value[Region], length: Code[Int]): Code[Long] = ???

  def forEach(mb: EmitMethodBuilder[_], addr: Code[Long], body: Code[Long] => Code[Unit]): Code[Unit] = ???

  def hasMissingValues(sourceOffset: Code[Long]): Code[Boolean] = ???

  def nextElementAddress(currentOffset: Long): Long = ???

  def nextElementAddress(currentOffset: Code[Long]): Code[Long] = ???

  def sType: SType = ???

  def unsafeOrdering(): UnsafeOrdering = ???

  def _asIdent: String = ???

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = ???

  def setRequired(required: Boolean): PType = ???

  protected[physical] def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = ???

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): SCode = ???

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = ???

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = ???

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = ???

  def loadFromNested(addr: Code[Long]): Code[Long] = ???

  def unstagedLoadFromNested(addr: Long): Long = ???

  def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = ???

  def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = ???
}
