package is.hail.annotations

import sun.misc.Unsafe

object Memory {
  private var unsafe = {
    val unsafeField = classOf[Unsafe].getDeclaredField("theUnsafe")
    unsafeField.setAccessible(true)
    unsafeField.get(null).asInstanceOf[Unsafe]
  }

  def storeByte(mem: Array[Byte], off: Long, b: Byte): Unit = unsafe.putByte(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, b)

  def storeFloat(mem: Array[Byte], off: Long, f: Float): Unit = unsafe.putFloat(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, f)

  def storeDouble(mem: Array[Byte], off: Long, d: Double): Unit = unsafe.putDouble(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off, d)

  def loadByte(mem: Array[Byte], off: Long): Byte = unsafe.getByte(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off)

  def loadFloat(mem: Array[Byte], off: Long): Float = unsafe.getFloat(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off)

  def loadDouble(mem: Array[Byte], off: Long): Double = unsafe.getDouble(mem, Unsafe.ARRAY_BYTE_BASE_OFFSET + off)

  def memcpy(dst: Array[Byte], dstOff: Long, src: Array[Byte], srcOff: Long, n: Long): Unit =
    unsafe.copyMemory(src, Unsafe.ARRAY_BYTE_BASE_OFFSET + srcOff, dst, Unsafe.ARRAY_BYTE_BASE_OFFSET + dstOff, n)

  def memcpy(dst: Long, src: Array[Byte], srcOff: Long, n: Long): Unit =
    copyFromArray(dst, src, srcOff, n)

  def memcpy(dst: Array[Byte], dstOff: Long, src: Long, n: Long): Unit =
    copyToArray(dst, dstOff, src, n)

  def loadBoolean(addr: Long): Boolean = unsafe.getByte(addr) != 0

  def loadByte(addr: Long): Byte = unsafe.getByte(addr)

  def loadShort(addr: Long): Short = unsafe.getShort(addr)

  def loadInt(addr: Long): Int = unsafe.getInt(addr)

  def loadLong(addr: Long): Long = unsafe.getLong(addr)

  def loadFloat(addr: Long): Float = unsafe.getFloat(addr)

  def loadDouble(addr: Long): Double = unsafe.getDouble(addr)

  def loadAddress(addr: Long): Long = unsafe.getAddress(addr)

  def storeBoolean(addr: Long, b: Boolean): Unit = unsafe.putByte(addr, if (b) 1 else 0)

  def storeByte(addr: Long, b: Byte): Unit = unsafe.putByte(addr, b)

  def storeShort(addr: Long, s: Short): Unit = unsafe.putShort(addr, s)

  def storeInt(addr: Long, i: Int): Unit = unsafe.putInt(addr, i)

  def storeLong(addr: Long, l: Long): Unit = unsafe.putLong(addr, l)

  def storeFloat(addr: Long, f: Float): Unit = unsafe.putFloat(addr, f)

  def storeDouble(addr: Long, d: Double): Unit = unsafe.putDouble(addr, d)

  def storeAddress(addr: Long, a: Long): Unit = unsafe.putAddress(addr, a)

  def malloc(size: Long): Long = unsafe.allocateMemory(size)

  def free(a: Long): Unit = unsafe.freeMemory(a)

  def realloc(a: Long, newSize: Long): Long = unsafe.reallocateMemory(a, newSize)

  def memcpy(dst: Long, src: Long, n: Long): Unit = unsafe.copyMemory(src, dst, n)

  def copyToArray(dst: Array[Byte], dstOff: Long, src: Long, n: Long): Unit = {
    unsafe.copyMemory(null, src, dst, Unsafe.ARRAY_BYTE_BASE_OFFSET + dstOff, n)
  }

  def copyFromArray(dst: Long, src: Array[Byte], srcOff: Long, n: Long): Unit = {
    unsafe.copyMemory(src, Unsafe.ARRAY_BYTE_BASE_OFFSET + srcOff, null, dst, n)
  }
}
