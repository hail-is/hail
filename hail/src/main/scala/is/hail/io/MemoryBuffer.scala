package is.hail.io

import is.hail.annotations.Memory

import java.util

final class MemoryBuffer extends Serializable {
  var mem: Array[Byte] = new Array[Byte](8)
  var pos: Int = 0
  var end: Int = 0

  def capacity: Int = mem.length

  def invalidate(): Unit =
    mem = null

  def clear(): Unit = {
    pos = 0
    end = 0
  }

  def clearPos(): Unit = {
    pos = 0
  }

  def set(bytes: Array[Byte]): Unit = {
    mem = bytes
    pos = 0
    end = bytes.length
  }

  def toByteArray(): Array[Byte] = {
    val dst = new Array[Byte](end)
    System.arraycopy(mem, 0, dst, 0, end);
    dst
  }

  def grow(n: Int): Unit = {
    mem = util.Arrays.copyOf(mem, math.max(capacity * 2, end + n))
  }

  def copyFrom(src: MemoryBuffer): Unit = {
    mem = util.Arrays.copyOf(src.mem, src.capacity)
    end = src.end
    pos = src.pos
  }

  def writeByte(b: Byte): Unit = {
    if (end + 1 > capacity)
      grow(1)
    Memory.storeByte(mem, end, b)
    end += 1
  }

  def writeInt(i: Int): Unit = {
    if (end + 4 > capacity)
      grow(4)
    Memory.storeInt(mem, end, i)
    end += 4
  }

  def writeLong(i: Long): Unit = {
    if (end + 8 > capacity)
      grow(8)
    Memory.storeLong(mem, end, i)
    end += 8
  }

  def writeFloat(i: Float): Unit = {
    if (end + 4 > capacity)
      grow(4)
    Memory.storeFloat(mem, end, i)
    end += 4
  }

  def writeDouble(i: Double): Unit = {
    if (end + 8 > capacity)
      grow(8)
    Memory.storeDouble(mem, end, i)
    end += 8
  }

  def writeBytes(off: Long, n: Int): Unit = {
    if (end + n > capacity)
      grow(n)
    Memory.memcpy(mem, end, off, n)
    end += n
  }

  def readByte(): Byte = {
    assert(pos + 1 <= end)
    val b = Memory.loadByte(mem, pos)
    pos += 1
    b
  }

  def readInt(): Int = {
    assert(pos + 4 <= end)
    val i = Memory.loadInt(mem, pos)
    pos += 4
    i
  }

  def readLong(): Long = {
    assert(pos + 8 <= end)
    val l = Memory.loadLong(mem, pos)
    pos += 8
    l
  }

  def readFloat(): Float = {
    assert(pos + 4 <= end)
    val f = Memory.loadFloat(mem, pos)
    pos += 4
    f
  }

  def readDouble(): Double = {
    assert(pos + 8 <= end)
    val d = Memory.loadDouble(mem, pos)
    pos += 8
    d
  }

  def readBytes(toOff: Long, n: Int): Unit = {
    assert(pos + n <= end)
    Memory.memcpy(toOff, mem, pos, n)
    pos += n
  }

  def readBytesArray(dst: Array[Byte], n: Int): Unit = {
    assert(pos + n <= end)
    System.arraycopy(mem, pos, dst, 0, n);
    pos += n
  }

  def skipByte(): Unit = {
    assert(pos + 1 <= end)
    pos += 1
  }

  def skipInt(): Unit = {
    assert(pos + 4 <= end)
    pos += 4
  }

  def skipLong(): Unit = {
    assert(pos + 8 <= end)
    pos += 8
  }

  def skipFloat(): Unit = {
    assert(pos + 4 <= end)
    pos += 4
  }

  def skipDouble(): Unit = {
    assert(pos + 8 <= end)
    pos += 8
  }

  def skipBytes(n: Int): Unit = {
    assert(pos + n <= end)
    pos += n
  }

  def dumpHexBytes(from: Int = 0, to: Int = end): Unit = {
    val bytes = (from until to).map { i =>
      val x = (mem(i).toInt & 0xff).toHexString
      if (x.length == 1) "0" + x
      else x
    }.mkString(" ")

    val index = (from until to by 4).map(i => String.format("%1$-12s", i.toString)).mkString("")
    println(s"bytes: $bytes")
    println(s"index: $index")
  }
}
