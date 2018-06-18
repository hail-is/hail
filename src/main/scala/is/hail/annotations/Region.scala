package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr.types._
import is.hail.utils._
import is.hail.nativecode._
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object Region {
  def apply(sizeHint: Long = 128): Region = {
    new Region()
  }

  def scoped[T](f: Region => T): T =
    using(Region())(f)
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

final class Region() extends NativeBase() {
  @native def nativeCtor(): Unit
  nativeCtor()
  
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
  
  final def align(a: Long) = nativeAlign(a)
  final def allocate(a: Long, n: Long): Long = nativeAlignAllocate(a, n)
  final def allocate(n: Long): Long = nativeAllocate(n)
  
  final def loadInt(addr: Long): Int = Memory.loadInt(addr)
  final def loadLong(addr: Long): Long = Memory.loadLong(addr)
  final def loadFloat(addr: Long): Float = Memory.loadFloat(addr)
  final def loadDouble(addr: Long): Double = Memory.loadDouble(addr)
  final def loadAddress(addr: Long): Long = Memory.loadLong(addr)
  final def loadByte(addr: Long): Byte = Memory.loadByte(addr)
  
  final def storeInt(addr: Long, v: Int) = Memory.storeInt(addr, v)
  final def storeLong(addr: Long, v: Long) = Memory.storeLong(addr, v)
  final def storeFloat(addr: Long, v: Float) = Memory.storeFloat(addr, v)
  final def storeDouble(addr: Long, v: Double) = Memory.storeDouble(addr, v)
  final def storeAddress(addr: Long, v: Long) = Memory.storeAddress(addr, v)
  final def storeByte(addr: Long, v: Byte) = Memory.storeByte(addr, v)
  
  final def loadBoolean(addr: Long): Boolean = if (Memory.loadByte(addr) == 0) false else true
  final def storeBoolean(addr: Long, v: Boolean) = Memory.storeByte(addr, if (v) 1 else 0)

  final def loadBytes(addr: Long, n: Int): Array[Byte] = {
    val a = new Array[Byte](n)
    Memory.copyToArray(a, 0, addr, n)
    a
  }

  final def loadBytes(addr: Long, dst: Array[Byte], dstOff: Long, n: Long): Unit = {
    Memory.copyToArray(dst, dstOff, addr, n)
  }

  final def storeBytes(addr: Long, src: Array[Byte]) {
    Memory.copyFromArray(addr, src, 0, src.length)
  }

  final def storeBytes(addr: Long, src: Array[Byte], srcOff: Long, n: Long) {
    Memory.copyFromArray(addr, src, srcOff, n)
  }

  final def copyFrom(src: Region, srcOff: Long, dstOff: Long, n: Long) {
    Memory.memcpy(dstOff, srcOff, n)
  }

  final def loadBit(byteOff: Long, bitOff: Long): Boolean = {
    val b = byteOff + (bitOff >> 3)
    (loadByte(b) & (1 << (bitOff & 7))) != 0
  }

  final def setBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) | (1 << (bitOff & 7))).toByte)
  }

  final def clearBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) & ~(1 << (bitOff & 7))).toByte)
  }

  final def storeBit(byteOff: Long, bitOff: Long, b: Boolean) {
    if (b)
      setBit(byteOff, bitOff)
    else
      clearBit(byteOff, bitOff)
  }

  final def appendBinary(v: Array[Byte]): Long = {
    val len: Int = v.length
    val grain = if (TBinary.contentAlignment < 4) 4 else TBinary.contentAlignment
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
    val grain = if (TBinary.contentAlignment < 4) 4 else TBinary.contentAlignment
    val addr = allocate(grain, grain+len) + (grain-4)
    storeInt(addr, len)
    copyFrom(fromRegion, TBinary.bytesOffset(fromOff) + start, addr+4, len)
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

  def visit(t: Type, off: Long, v: ValueVisitor) {
    t match {
      case _: TBoolean => v.visitBoolean(loadBoolean(off))
      case _: TInt32 => v.visitInt32(loadInt(off))
      case _: TInt64 => v.visitInt64(loadLong(off))
      case _: TFloat32 => v.visitFloat32(loadFloat(off))
      case _: TFloat64 => v.visitFloat64(loadDouble(off))
      case _: TString =>
        val boff = off
        v.visitString(TString.loadString(this, boff))
      case _: TBinary =>
        val boff = off
        val length = TBinary.loadLength(this, boff)
        val b = loadBytes(TBinary.bytesOffset(boff), length)
        v.visitBinary(b)
      case t: TContainer =>
        val aoff = off
        val length = t.loadLength(this, aoff)
        v.enterArray(t, length)
        var i = 0
        while (i < length) {
          v.enterElement(i)
          if (t.isElementDefined(this, aoff, i))
            visit(t.elementType, t.loadElement(this, aoff, length, i), v)
          else
            v.visitMissing(t.elementType)
          i += 1
        }
        v.leaveArray()
      case t: TStruct =>
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
      case t: TTuple =>
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
      case t: ComplexType =>
        visit(t.representation, off, v)
    }
  }

  def pretty(t: Type, off: Long): String = {
    val v = new PrettyVisitor()
    visit(t, off, v)
    v.result()
  }
  
  def prettyBits(): String = {
    "FIXME: implement prettyBits on Region"
  }

}
