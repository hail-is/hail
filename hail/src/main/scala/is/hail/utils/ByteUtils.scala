package is.hail.utils

import java.io.{ ByteArrayOutputStream, InputStream, OutputStream }
import java.nio.ByteBuffer
import java.nio.charset.{ Charset, StandardCharsets }

object ByteUtils {
  def writeInt(out: OutputStream, i: Int): Unit = {
    out.write(i)
    out.write(i >> 8)
    out.write(i >> 16)
    out.write(i >> 24)
  }

  def writeInt(out: Array[Byte], off: Int, i: Int): Int = {
    out(off) = (i & 0xff).toByte
    out(off + 1) = ((i >> 8) & 0xff).toByte
    out(off + 2) = ((i >> 16) & 0xff).toByte
    out(off + 3) = ((i >> 24) & 0xff).toByte
    off + 4
  }

  def readInt(in: InputStream): Int = {
    var out = 0
    var i = in.read()
    assert(i != -1)
    out |= (i & 0xff)
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 8
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 16
    i = in.read()
    assert(i != -1)
    out | (i & 0xff) << 24
  }

  def skipInt(in: ByteBuffer): Unit = {
    in.get()
    in.get()
    in.get()
    in.get()
  }

  def readInt(in: ByteBuffer): Int = {
    var out = 0
    var i = in.get()
    out |= (i & 0xff)
    i = in.get()
    out |= (i & 0xff) << 8
    i = in.get()
    out |= (i & 0xff) << 16
    i = in.get()
    out | (i & 0xff) << 24
  }

  def writeLong(out: OutputStream, l: Long): Unit = {
    out.write(l.toInt)
    out.write((l >> 8).toInt)
    out.write((l >> 16).toInt)
    out.write((l >> 24).toInt)
    out.write((l >> 32).toInt)
    out.write((l >> 40).toInt)
    out.write((l >> 48).toInt)
    out.write((l >> 56).toInt)
  }

  def writeLong(out: Array[Byte], off: Int, l: Long): Int = {
    out(off) = (l & 0xff).toByte
    out(off + 1) = ((l >> 8) & 0xff).toByte
    out(off + 2) = ((l >> 16) & 0xff).toByte
    out(off + 3) = ((l >> 24) & 0xff).toByte
    out(off + 4) = ((l >> 32) & 0xff).toByte
    out(off + 5) = ((l >> 40) & 0xff).toByte
    out(off + 6) = ((l >> 48) & 0xff).toByte
    out(off + 7) = ((l >> 56) & 0xff).toByte
    off + 8
  }

  def readLong(in: InputStream): Long = {
    var out = 0L
    var i = in.read()
    assert(i != -1)
    out |= (i & 0xff)
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 8
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 16
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 24
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 32
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 40
    i = in.read()
    assert(i != -1)
    out |= (i & 0xff) << 48
    i = in.read()
    assert(i != -1)
    out | (i & 0xff) << 56
  }

  def readLong(in: ByteBuffer): Long = {
    var out = 0L
    var i = in.get()
    out |= (i & 0xff)
    i = in.get()
    out |= (i & 0xff) << 8
    i = in.get()
    out |= (i & 0xff) << 16
    i = in.get()
    out |= (i & 0xff) << 24
    i = in.get()
    out |= (i & 0xff) << 32
    i = in.get()
    out |= (i & 0xff) << 40
    i = in.get()
    out |= (i & 0xff) << 48
    i = in.get()
    out | (i & 0xff) << 56
  }

  def readBytes(in: InputStream, n: Int): Array[Byte] = {
    val data = new Array[Byte](n)
    var offset = 0
    var bytesRead = 0
    bytesRead = in.read(data, offset, n - offset)
    while (bytesRead > 0) {
      offset += bytesRead
      bytesRead = in.read(data, offset, n - offset)
    }
    assert(bytesRead <= 0)
    data
  }

  def readBytes(in: ByteBuffer, n: Int): Array[Byte] = {
    val data = new Array[Byte](n)
    in.get(data)
    data
  }

  def writeByteArray(out: OutputStream, bytes: Array[Byte]): Unit = {
    writeInt(out, bytes.length)
    out.write(bytes)
  }

  def writeByteArray(out: Array[Byte], _off: Int, bytes: Array[Byte]): Int = {
    var off = _off
    off = writeInt(out, off, bytes.length)
    System.arraycopy(bytes, 0, out, off, bytes.length)
    off + bytes.length
  }

  def skipByteArray(in: InputStream): Unit = {
    val len = readInt(in)
    in.skip(len)
  }

  def readByteArray(in: InputStream): Array[Byte] = {
    val len = readInt(in)
    readBytes(in, len)
  }

  def readByteArray(in: ByteBuffer): Array[Byte] = {
    val len = readInt(in)
    readBytes(in, len)
  }

  def stringToBytes(s: String, charset: Charset = StandardCharsets.UTF_8): Array[Byte] =
    s.getBytes(charset)

  def writeString(
    out: OutputStream,
    s: String,
    charset: Charset = StandardCharsets.UTF_8
  ): Unit =
    writeByteArray(out, stringToBytes(s))

  def readString(
    in: ByteBuffer,
    charset: Charset = StandardCharsets.UTF_8
  ): String =
    new String(readByteArray(in), charset)
}
