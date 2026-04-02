package is.hail.io

import is.hail.annotations.Memory

import java.nio.ByteOrder

object PrefixCoder {
  val PLATFORM_IS_BE: Boolean = ByteOrder.nativeOrder == ByteOrder.BIG_ENDIAN

  def swap(v: Int): Int = if (PLATFORM_IS_BE) v else java.lang.Integer.reverseBytes(v)

  def swap(v: Long): Long = if (PLATFORM_IS_BE) v else java.lang.Long.reverseBytes(v)

  // flips the sign bit, this + writing integer values as big endian gives
  // a series of bytes where memcmp agrees with natural integer ordering
  def flip(v: Int): Int = v ^ (1 << 31)

  def flip(v: Long): Long = v ^ (1L << 63)
}

class PrefixCoder() {
  import PrefixCoder._

  val mb = new MemoryBuffer

  private val scratch = new Array[Byte](8)

  def clear() = mb.clear()

  def toByteArray(): Array[Byte] = mb.toByteArray()

  def encodeBool(b: Boolean) = mb.writeByte(if (b) 1 else 0)

  def encodeInt(v: Int) = mb.writeInt(swap(flip(v)))

  def encodeLong(v: Long) = mb.writeLong(swap(flip(v)))

  // This produces a total ordering on floats that is close to the expected natural ordering:
  // -qNaN < -sNaN < -Inf < -normal < -subnormal < -0 < 0 < +subnormal < +normal < +Inf < +sNaN < +qNaN
  def encodeFloat(v: Float) = {
    Memory.storeFloat(scratch, 0, v);
    val bits = Memory.loadInt(scratch, 0);
    val i = if (bits < 0) ~bits else flip(bits)
    mb.writeInt(swap(i))
  }

  def encodeDouble(v: Double) = {
    Memory.storeDouble(scratch, 0, v);
    val bits = Memory.loadLong(scratch, 0);
    val i = if (bits < 0) ~bits else flip(bits)
    mb.writeLong(swap(i))
  }

  // in hail, missing sorts after present
  def encodeMissing() = encodeBool(true)
  def encodePresent() = encodeBool(false)

  // arrays are encoded with a continuation byte before each element or 0 for a terminator.
  def encodeTerminator() = encodeBool(false)
  def encodeContinuation() = encodeBool(true)

  // bytes like types have a double null terminator, null bytes are escaped into 0x00 0xFF
  def writeBytes(bs: Array[Byte]) = {
    var i = 0
    while (i < bs.length) {
      mb.writeByte(bs(i))
      if (bs(i) == 0) {
        mb.writeByte(0xFF.toByte)
      }
      i += 1
    }
    mb.writeByte(0.toByte);
    mb.writeByte(0.toByte);
  }
}
