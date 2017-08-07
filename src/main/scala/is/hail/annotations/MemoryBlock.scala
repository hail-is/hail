package is.hail.annotations

import java.util

import org.apache.spark.unsafe.Platform

final class MemoryBlock(val mem: Array[Long]) {
  require(mem.length < (Integer.MAX_VALUE / 8), "too big")

  def sizeInBytes: Int = mem.length * 8

  def copyFrom(other: MemoryBlock, readStart: Int, writeStart: Int, size: Int) {
    Platform.copyMemory(other.mem, readStart + Platform.LONG_ARRAY_OFFSET, mem,
      writeStart + Platform.LONG_ARRAY_OFFSET, size)
  }

  def loadInt(off: Int): Int = {
    assert(off + 4 <= sizeInBytes, s"tried to read int from offset $off with array size $sizeInBytes")
    Platform.getInt(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadLong(off: Int): Long = {
    assert(off + 8 <= sizeInBytes, s"tried to read long from offset $off with array size $sizeInBytes")
    Platform.getLong(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadFloat(off: Int): Float = {
    assert(off + 4 <= sizeInBytes, s"tried to read float from offset $off with array size $sizeInBytes")
    Platform.getFloat(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadDouble(off: Int): Double = {
    assert(off + 8 <= sizeInBytes, s"tried to read double from offset $off with array size $sizeInBytes")
    Platform.getDouble(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadByte(off: Int): Byte = {
    assert(off + 1 <= sizeInBytes, s"tried to read byte from offset $off with array size $sizeInBytes")
    Platform.getByte(mem, Platform.LONG_ARRAY_OFFSET + off)
  }

  def loadBytes(off: Int, size: Int): Array[Byte] = {
    assert(off + size <= sizeInBytes, s"tried to read bytes of size $size from offset $off with array size $sizeInBytes")
    val a = new Array[Byte](size)
    Platform.copyMemory(mem, Platform.LONG_ARRAY_OFFSET + off, a, Platform.BYTE_ARRAY_OFFSET, size)
    a
  }

  def storeInt(off: Int, i: Int) {
    assert(off + 4 <= sizeInBytes, s"tried to store int to offset $off with array size $sizeInBytes")
    Platform.putInt(mem, Platform.LONG_ARRAY_OFFSET + off, i)
  }

  def storeLong(off: Int, l: Long) {
    assert(off + 8 <= sizeInBytes, s"tried to store long to offset $off with array size $sizeInBytes")
    Platform.putLong(mem, Platform.LONG_ARRAY_OFFSET + off, l)
  }

  def storeFloat(off: Int, f: Float) {
    assert(off + 4 <= sizeInBytes, s"tried to store float to offset $off with array size $sizeInBytes")
    Platform.putFloat(mem, Platform.LONG_ARRAY_OFFSET + off, f)
  }

  def storeDouble(off: Int, d: Double) {
    assert(off + 8 <= sizeInBytes, s"tried to store double to offset $off with array size $sizeInBytes")
    Platform.putDouble(mem, Platform.LONG_ARRAY_OFFSET + off, d)
  }

  def storeByte(off: Int, b: Byte) {
    assert(off + 1 <= sizeInBytes, s"tried to store byte to offset $off with array size $sizeInBytes")
    Platform.putByte(mem, Platform.LONG_ARRAY_OFFSET + off, b)
  }

  def storeBytes(off: Int, bytes: Array[Byte]) {
    assert(off + bytes.length <= sizeInBytes, s"tried to store ${ bytes.length } bytes to offset $off with array size $sizeInBytes")
    Platform.copyMemory(bytes, Platform.BYTE_ARRAY_OFFSET, mem, Platform.LONG_ARRAY_OFFSET + off, bytes.length)
  }

  def reallocate(size: Int): MemoryBlock = {
    if (sizeInBytes < size) {
      val newMem = new Array[Long](math.max(mem.length * 2, (size + 7) / 8))
      Platform.copyMemory(mem, Platform.LONG_ARRAY_OFFSET, newMem, Platform.LONG_ARRAY_OFFSET, sizeInBytes)
      new MemoryBlock(newMem)
    } else
      this
  }

  def copy(): MemoryBlock = new MemoryBlock(util.Arrays.copyOf(mem, mem.length))
}

final class MemoryBuffer(sizeHint: Int = 128) {
  var mb = new MemoryBlock(new Array[Long]((sizeHint + 7) / 8))

  var offset: Int = 0

  def alignAndEnsure(size: Int) {
    align(size)
    ensure(size)
  }

  def ensure(size: Int) {
    mb = mb.reallocate(offset + size)
  }

  def loadInt(off: Int): Int = mb.loadInt(off)

  def loadLong(off: Int): Long = mb.loadLong(off)

  def loadFloat(off: Int): Float = mb.loadFloat(off)

  def loadDouble(off: Int): Double = mb.loadDouble(off)

  def loadByte(off: Int): Byte = mb.loadByte(off)

  def loadBytes(off: Int, size: Int): Array[Byte] = mb.loadBytes(off, size)

  def storeInt(off: Int, i: Int) {
    mb.storeInt(off, i)
  }

  def storeLong(off: Int, l: Long) {
    mb.storeLong(off, l)
  }

  def storeFloat(off: Int, f: Float) {
    mb.storeFloat(off, f)
  }

  def storeDouble(off: Int, d: Double) {
    mb.storeDouble(off, d)
  }

  def storeByte(off: Int, b: Byte) {
    mb.storeByte(off, b)
  }

  def storeBytes(off: Int, bytes: Array[Byte]) {
    mb.storeBytes(off, bytes)
  }

  def appendInt(i: Int) {
    alignAndEnsure(4)
    mb.storeInt(offset, i)
    offset += 4
  }

  def appendLong(l: Long) {
    alignAndEnsure(8)
    mb.storeLong(offset, l)
    offset += 8
  }

  def appendFloat(f: Float) {
    alignAndEnsure(4)
    mb.storeFloat(offset, f)
    offset += 4
  }

  def appendDouble(d: Double) {
    alignAndEnsure(8)
    mb.storeDouble(offset, d)
    offset += 8
  }

  def appendByte(b: Byte) {
    ensure(1)
    mb.storeByte(offset, b)
    offset += 1
  }

  def appendBytes(bytes: Array[Byte]) {
    ensure(bytes.length)
    mb.storeBytes(offset, bytes)
    offset += bytes.length
  }

  def allocate(nBytes: Int): Int = {
    val currentOffset = offset
    ensure(nBytes)
    offset += nBytes
    currentOffset
  }

  def align(alignment: Int) {
    assert(alignment > 0, s"invalid alignment: $alignment")
    assert((alignment & (alignment - 1)) == 0, s"invalid alignment: $alignment") // power of 2
    offset = (offset + (alignment - 1)) & ~(alignment - 1)
  }

  def copyFrom(other: MemoryBlock, readStart: Int, writeStart: Int, size: Int) {
    assert(writeStart <= (offset - size))
    mb.copyFrom(other, readStart, writeStart, size)
  }

  def clear() {
    offset = 0
  }

  def result(): MemoryBlock = {
    val reqLength = (offset + 7) / 8
    val arr = new Array[Long](reqLength)
    Platform.copyMemory(mb.mem, Platform.LONG_ARRAY_OFFSET, arr, Platform.LONG_ARRAY_OFFSET, offset)
    new MemoryBlock(arr)
  }
}
