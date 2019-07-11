package is.hail.annotations

import is.hail.asm4s
import is.hail.asm4s.{Code, coerce}
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.nativecode._

object Region {
  val regular: Int = 64 * 1024
  val small: Int = 8 * 1024
  val tiny: Int = 1024

  def apply(sizeHint: Long = 128): Region = new Region()

  def scoped[T](f: Region => T): T =
    using(Region())(f)

  def smallScoped[T](f: Region => T): T = using(new Region(8 * 1024))(f)

  def tinyScoped[T](f: Region => T): T = using(new Region(1024))(f)
  
  def loadInt(addr: Long): Int = Memory.loadInt(addr)
  def loadLong(addr: Long): Long = Memory.loadLong(addr)
  def loadFloat(addr: Long): Float = Memory.loadFloat(addr)
  def loadDouble(addr: Long): Double = Memory.loadDouble(addr)
  def loadAddress(addr: Long): Long = Memory.loadLong(addr)
  def loadByte(addr: Long): Byte = Memory.loadByte(addr)

  def storeInt(addr: Long, v: Int): Unit = Memory.storeInt(addr, v)
  def storeLong(addr: Long, v: Long): Unit = Memory.storeLong(addr, v)
  def storeFloat(addr: Long, v: Float): Unit = Memory.storeFloat(addr, v)
  def storeDouble(addr: Long, v: Double): Unit = Memory.storeDouble(addr, v)
  def storeAddress(addr: Long, v: Long): Unit = Memory.storeAddress(addr, v)
  def storeByte(addr: Long, v: Byte): Unit = Memory.storeByte(addr, v)

  def loadBoolean(addr: Long): Boolean = if (Memory.loadByte(addr) == 0) false else true
  def storeBoolean(addr: Long, v: Boolean): Unit = Memory.storeByte(addr, if (v) 1 else 0)

  def loadBytes(addr: Long, n: Int): Array[Byte] = {
    val a = new Array[Byte](n)
    Memory.copyToArray(a, 0, addr, n)
    a
  }

  def loadBytes(addr: Long, dst: Array[Byte], dstOff: Long, n: Long): Unit = {
    Memory.copyToArray(dst, dstOff, addr, n)
  }

  def storeBytes(addr: Long, src: Array[Byte]) {
    Memory.copyFromArray(addr, src, 0, src.length)
  }

  def storeBytes(addr: Long, src: Array[Byte], srcOff: Long, n: Long) {
    Memory.copyFromArray(addr, src, srcOff, n)
  }

  def copyFrom(srcOff: Long, dstOff: Long, n: Long) {
    Memory.memcpy(dstOff, srcOff, n)
  }

  def loadBit(byteOff: Long, bitOff: Long): Boolean = {
    val b = byteOff + (bitOff >> 3)
    (loadByte(b) & (1 << (bitOff & 7))) != 0
  }

  def setBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) | (1 << (bitOff & 7))).toByte)
  }

  def clearBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) & ~(1 << (bitOff & 7))).toByte)
  }

  def storeBit(byteOff: Long, bitOff: Long, b: Boolean) {
    if (b)
      setBit(byteOff, bitOff)
    else
      clearBit(byteOff, bitOff)
  }


  def loadInt(addr: Code[Long]): Code[Int] = Code.invokeScalaObject[Long, Int](Region.getClass, "loadInt", addr)
  def loadLong(addr: Code[Long]): Code[Long] = Code.invokeScalaObject[Long, Long](Region.getClass, "loadLong", addr)
  def loadFloat(addr: Code[Long]): Code[Float] = Code.invokeScalaObject[Long, Float](Region.getClass, "loadFloat", addr)
  def loadDouble(addr: Code[Long]): Code[Double] = Code.invokeScalaObject[Long, Double](Region.getClass, "loadDouble", addr)
  def loadAddress(addr: Code[Long]): Code[Long] = Code.invokeScalaObject[Long, Long](Region.getClass, "loadAddress", addr)
  def loadByte(addr: Code[Long]): Code[Byte] = Code.invokeScalaObject[Long, Byte](Region.getClass, "loadByte", addr)

  def storeInt(addr: Code[Long], v: Code[Int]): Code[Unit] = Code.invokeScalaObject[Long, Int, Unit](Region.getClass, "storeInt", addr, v)
  def storeLong(addr: Code[Long], v: Code[Long]): Code[Unit] = Code.invokeScalaObject[Long, Long, Unit](Region.getClass, "storeLong", addr, v)
  def storeFloat(addr: Code[Long], v: Code[Float]): Code[Unit] = Code.invokeScalaObject[Long, Float, Unit](Region.getClass, "storeFloat", addr, v)
  def storeDouble(addr: Code[Long], v: Code[Double]): Code[Unit] = Code.invokeScalaObject[Long, Double, Unit](Region.getClass, "storeDouble", addr, v)
  def storeAddress(addr: Code[Long], v: Code[Long]): Code[Unit] = Code.invokeScalaObject[Long, Long, Unit](Region.getClass, "storeAddress", addr, v)
  def storeByte(addr: Code[Long], v: Code[Byte]): Code[Unit] = Code.invokeScalaObject[Long, Byte, Unit](Region.getClass, "storeByte", addr, v)

  def loadBoolean(addr: Code[Long]): Code[Boolean] = Code.invokeScalaObject[Long, Boolean](Region.getClass, "loadBoolean", addr)
  def storeBoolean(addr: Code[Long], v: Code[Boolean]): Code[Unit] = Code.invokeScalaObject[Long, Boolean, Unit](Region.getClass, "storeBoolean", addr, v)
  def loadBytes(addr: Code[Long], n: Code[Int]): Code[Array[Byte]] = Code.invokeScalaObject[Long, Int, Array[Byte]](Region.getClass, "loadBytes", addr, n)
  def loadBytes(addr: Code[Long], dst: Code[Array[Byte]], dstOff: Code[Long], n: Code[Long]): Unit =
    Code.invokeScalaObject[Long, Array[Byte], Long, Long, Unit](Region.getClass, "loadBytes", addr, dst, dstOff, n)
  def storeBytes(addr: Code[Long], src: Code[Array[Byte]]): Code[Unit] = Code.invokeScalaObject[Long, Array[Byte], Unit](Region.getClass, "storeBytes", addr, src)
  def storeBytes(addr: Code[Long], src: Code[Array[Byte]], srcOff: Code[Long], n: Code[Long]): Code[Unit] =
    Code.invokeScalaObject[Long, Array[Byte], Long, Long, Unit](Region.getClass, "storeBytes", addr, src, srcOff, n)

  def copyFrom(srcOff: Code[Long], dstOff: Code[Long], n: Code[Long]): Code[Unit] =
    Code.invokeScalaObject[Long, Long, Long, Unit](Region.getClass, "copyFrom", srcOff, dstOff, n)
  def loadBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Boolean] =
    Code.invokeScalaObject[Long, Long, Boolean](Region.getClass, "loadBit", byteOff, bitOff)
  def setBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] =
    Code.invokeScalaObject[Long, Long, Unit](Region.getClass, "setBit", byteOff, bitOff)
  def clearBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] =
    Code.invokeScalaObject[Long, Long, Unit](Region.getClass, "clearBit", byteOff, bitOff)
  def storeBit(byteOff: Code[Long], bitOff: Code[Long], b: Code[Boolean]): Code[Unit] =
    Code.invokeScalaObject[Long, Long, Boolean, Unit](Region.getClass, "storeBit", byteOff, bitOff, b)

  def loadPrimitive(typ: PType): Code[Long] => Code[_] = typ.fundamentalType match {
    case _: PBoolean => loadBoolean
    case _: PInt32 => loadInt
    case _: PInt64 => loadLong
    case _: PFloat32 => loadFloat
    case _: PFloat64 => loadDouble
  }

  def storePrimitive(typ: PType, dest: Code[Long]): Code[_] => Code[Unit] = typ.fundamentalType match {
    case _: PBoolean => v => storeBoolean(dest, coerce[Boolean](v))
    case _: PInt32 => v => storeInt(dest, coerce[Int](v))
    case _: PInt64 => v => storeLong(dest, coerce[Long](v))
    case _: PFloat32 => v => storeFloat(dest, coerce[Float](v))
    case _: PFloat64 => v => storeDouble(dest, coerce[Double](v))
  }
}

// Off-heap implementation of Region differs from the previous
// Scala contiguous-buffer implementation.
//
//  - it now has multiple memory chunks rather than one contiguous buffer
//
//  - references are now absolute addresses rather than buffer-offsets
//
//  - consequently, Region is not copy'able or Serializable, because
//    those operations have to know the RegionValue's Type to convert
//    within-Region references to/from absolute addresses.

final class Region private (empty: Boolean) extends NativeBase() {
  def this(blockSize: Int = 64 * 1024) {
    this(false)
    nativeCtor(RegionPool.get, blockSize)
  }
  @native def nativeCtor(p: RegionPool, blockSize: Int): Unit
  @native def initEmpty(): Unit
  @native def nativeClearRegion(): Unit

  if (empty) initEmpty()
  
  def this(b: Region) {
    this()
    copyAssign(b)
  }  

  // FIXME: not sure what this should mean ...
  // def setFrom(b: Region) { }

  final def copyAssign(b: Region) = super.copyAssign(b)
  final def moveAssign(b: Region) = super.moveAssign(b)
  
  @native def clearButKeepMem(): Unit
  final def clear(): Unit = clearButKeepMem()
  
  @native def nativeAlign(alignment: Long): Unit
  @native def nativeAlignAllocate(alignment: Long, n: Long): Long
  @native def nativeAllocate(n: Long): Long
  @native def nativeReference(r2: Region): Unit
  @native def nativeRefreshRegion(): Unit

  @native def nativeGetNumParents(): Int
  @native def nativeSetNumParents(n: Int): Unit
  @native def nativeSetParentReference(r2: Region, i: Int): Unit
  @native def nativeGetParentReferenceInto(r2: Region, i: Int, blockSize: Int): Region
  @native def nativeClearParentReference(i: Int): Unit

  final def align(a: Long) = nativeAlign(a)
  final def allocate(a: Long, n: Long): Long = nativeAlignAllocate(a, n)
  final def allocate(n: Long): Long = nativeAllocate(n)

  // FIXME: using nativeGetNumParents for now because we're not using `reference` in Scala
//  private var explicitParents: Int = 0

  final def reference(other: Region): Unit = {
//    assert(explicitParents <= 0, s"can't use 'reference' if you're explicitly setting Region dependencies")
//    explicitParents = -1
    nativeReference(other)
  }

  final def refreshRegion(): Unit = nativeRefreshRegion()

  def setNumParents(n: Int): Unit = {
    assert(nativeGetNumParents() >= 0 && nativeGetNumParents() <= n)
    nativeSetNumParents(n)
  }

  def setParentReference(r: Region, i: Int): Unit = {
    assert(i < nativeGetNumParents())
    nativeSetParentReference(r, i)
  }

  def getParentReference(i: Int, blockSize: Int): Region = {
    assert(i < nativeGetNumParents())
    val r = new Region(empty = true)
    nativeGetParentReferenceInto(r, i, blockSize)
    r
  }

  def clearParentReference(i: Int): Unit = {
    assert(i < nativeGetNumParents())
    nativeClearParentReference(i)
  }
  
  final def loadInt(addr: Long): Int = Region.loadInt(addr)
  final def loadLong(addr: Long): Long = Region.loadLong(addr)
  final def loadFloat(addr: Long): Float = Region.loadFloat(addr)
  final def loadDouble(addr: Long): Double = Region.loadDouble(addr)
  final def loadAddress(addr: Long): Long = Region.loadLong(addr)
  final def loadByte(addr: Long): Byte = Region.loadByte(addr)
  
  final def storeInt(addr: Long, v: Int): Unit = Region.storeInt(addr, v)
  final def storeLong(addr: Long, v: Long): Unit = Region.storeLong(addr, v)
  final def storeFloat(addr: Long, v: Float): Unit = Region.storeFloat(addr, v)
  final def storeDouble(addr: Long, v: Double): Unit = Region.storeDouble(addr, v)
  final def storeAddress(addr: Long, v: Long): Unit = Region.storeAddress(addr, v)
  final def storeByte(addr: Long, v: Byte): Unit = Region.storeByte(addr, v)
  
  final def loadBoolean(addr: Long): Boolean = Region.loadBoolean(addr)
  final def storeBoolean(addr: Long, v: Boolean): Unit = Region.storeBoolean(addr, v)

  final def loadBytes(addr: Long, n: Int): Array[Byte] = Region.loadBytes(addr, n)
  final def loadBytes(addr: Long, dst: Array[Byte], dstOff: Long, n: Long): Unit =
    Region.loadBytes(addr, dst, dstOff, n)

  final def storeBytes(addr: Long, src: Array[Byte]): Unit = Region.storeBytes(addr, src)
  final def storeBytes(addr: Long, src: Array[Byte], srcOff: Long, n: Long): Unit =
    Region.storeBytes(addr, src, srcOff, n)
  final def copyFrom(src: Region, srcOff: Long, dstOff: Long, n: Long) =
    Region.copyFrom(srcOff, dstOff, n)

  final def loadBit(byteOff: Long, bitOff: Long): Boolean = Region.loadBit(byteOff, bitOff)
  final def setBit(byteOff: Long, bitOff: Long): Unit = Region.setBit(byteOff, bitOff)
  final def clearBit(byteOff: Long, bitOff: Long): Unit = Region.clearBit(byteOff, bitOff)
  final def storeBit(byteOff: Long, bitOff: Long, b: Boolean): Unit = Region.storeBit(byteOff, bitOff, b)

  final def appendBinary(v: Array[Byte]): Long = {
    val len: Int = v.length
    val grain = if (PBinary.contentAlignment < 4) 4 else PBinary.contentAlignment
    val addr = allocate(grain, grain+len) + (grain-4)
    storeInt(addr, len)
    storeBytes(addr+4, v)
    addr
  }
  
  final def appendBinarySlice(
    fromRegion: Region,
    fromOff: Long,
    start: Int,
    len: Int
  ): Long = {
    assert(len >= 0)
    val grain = if (PBinary.contentAlignment < 4) 4 else PBinary.contentAlignment
    val addr = allocate(grain, grain+len) + (grain-4)
    storeInt(addr, len)
    copyFrom(fromRegion, PBinary.bytesOffset(fromOff) + start, addr+4, len)
    addr
  }

  // Use of appendXXX methods is deprecated now that Region uses absolute
  // addresses and non-contiguous memory allocation.  You can't assume any
  // relationships between the addresses returned by appendXXX methods -
  // and to make it even more confusing, there may be long sequences of 
  // ascending addresses (within a buffer) followed by an arbitrary jump
  // to an address in a different buffer.
  
  final def appendArrayInt(v: Array[Int]): Long = {
    val len: Int = v.length
    val addr = allocate(4, 4*(1+len))
    storeInt(addr, len)
    val data = addr+4
    var idx = 0
    while (idx < len) {
      storeInt(data + 4 * idx, v(idx))
      idx += 1
    }
    addr
  }
  
  final def appendInt(v: Int): Long = {
    val a = allocate(4, 4)
    Memory.storeInt(a, v)
    a
  }
  final def appendLong(v: Long): Long = {
    val a = allocate(8, 8)
    Memory.storeLong(a, v)
    a
  }
  final def appendFloat(v: Float): Long = {
    val a = allocate(4, 4)
    Memory.storeFloat(a, v)
    a
  }
  final def appendDouble(v: Double): Long = {
    val a = allocate(8, 8)
    Memory.storeDouble(a, v)
    a
  }
  final def appendByte(v: Byte): Long = {
    val a = allocate(1)
    Memory.storeByte(a, v)
    a
  }
  final def appendString(v: String): Long =
    appendBinary(v.getBytes)
  
  final def appendStringSlice(fromRegion: Region, fromOff: Long, start: Int, n: Int): Long =
    appendBinarySlice(fromRegion, fromOff, start, n)

  def visit(t: PType, off: Long, v: ValueVisitor) {
    t match {
      case _: PBoolean => v.visitBoolean(loadBoolean(off))
      case _: PInt32 => v.visitInt32(loadInt(off))
      case _: PInt64 => v.visitInt64(loadLong(off))
      case _: PFloat32 => v.visitFloat32(loadFloat(off))
      case _: PFloat64 => v.visitFloat64(loadDouble(off))
      case _: PString =>
        val boff = off
        v.visitString(PString.loadString(this, boff))
      case _: PBinary =>
        val boff = off
        val length = PBinary.loadLength(this, boff)
        val b = loadBytes(PBinary.bytesOffset(boff), length)
        v.visitBinary(b)
      case t: PContainer =>
        val aoff = off
        val pt = t
        val length = pt.loadLength(this, aoff)
        v.enterArray(t, length)
        var i = 0
        while (i < length) {
          v.enterElement(i)
          if (pt.isElementDefined(this, aoff, i))
            visit(t.elementType, pt.loadElement(this, aoff, length, i), v)
          else
            v.visitMissing(t.elementType)
          i += 1
        }
        v.leaveArray()
      case t: PStruct =>
        v.enterStruct(t)
        var i = 0
        while (i < t.size) {
          val f = t.fields(i)
          v.enterField(f)
          if (t.isFieldDefined(this, off, i))
            visit(f.typ, t.loadField(this, off, i), v)
          else
            v.visitMissing(f.typ)
          v.leaveField()
          i += 1
        }
        v.leaveStruct()
      case t: PTuple =>
        v.enterTuple(t)
        var i = 0
        while (i < t.size) {
          v.enterElement(i)
          if (t.isFieldDefined(this, off, i))
            visit(t.types(i), t.loadField(this, off, i), v)
          else
            v.visitMissing(t.types(i))
          v.leaveElement()
          i += 1
        }
        v.leaveTuple()
      case t: ComplexPType =>
        visit(t.representation, off, v)
    }
  }

  def pretty(t: PType, off: Long): String = {
    val v = new PrettyVisitor()
    visit(t, off, v)
    v.result()
  }
  
  def prettyBits(): String = {
    "FIXME: implement prettyBits on Region"
  }
}

object RegionPool {
  private val pools = new java.util.concurrent.ConcurrentHashMap[Long, RegionPool]()

  def get: RegionPool = {
    val makePool: java.util.function.Function[Long, RegionPool] = new java.util.function.Function[Long, RegionPool] {
      def apply(id: Long): RegionPool = new RegionPool()
    }
    pools.computeIfAbsent(Thread.currentThread().getId(), makePool)
  }
}

class RegionPool private() extends NativeBase() {
  @native def nativeCtor(): Unit
  nativeCtor()

  @native def numRegions(): Int
  @native def numFreeRegions(): Int
  @native def numFreeBlocks(): Int

  def logStats(context: String): Unit = {
    val pool = RegionPool.get
    val nFree = pool.numFreeRegions()
    val nRegions = pool.numRegions()
    val nBlocks = pool.numFreeBlocks()

    log.info(
      s"""Region count for $context
         |    regions: $nRegions
         |     blocks: $nBlocks
         |       free: $nFree
         |       used: ${ nRegions - nFree }""".stripMargin)
  }
}

object RegionUtils {
  def printBytes(off: Long, n: Int, header: String): String = {
    Region.loadBytes(off, n).zipWithIndex
      .grouped(16)
      .map(bs => bs.map { case (b, _) => "%02x".format(b) }.mkString("  %016x: ".format(off + bs(0)._2), " ", ""))
      .mkString(if (header != null) s"$header\n" else "\n", "\n", "")
  }

  def printBytes(off: Code[Long], n: Int, header: String): Code[String] =
    Code.invokeScalaObject[Long, Int, String, String](RegionUtils.getClass, "printBytes", off, n, asm4s.const(header))
}