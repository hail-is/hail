package is.hail.io

import is.hail.annotations.Memory
import is.hail.utils._

import java.io.{Closeable, InputStream, OutputStream}

final class DoubleInputBuffer(in: InputStream, bufSize: Int) extends Closeable {
  private val buf = new Array[Byte](bufSize)
  private var end: Int = 0
  private var off: Int = 0

  def close(): Unit = {
    in.close()
  }

  def readDoubles(to: Array[Double]): Unit = readDoubles(to, 0, to.length)

  def readDoubles(to: Array[Double], toOff0: Int, n0: Int): Unit = {
    assert(toOff0 >= 0)
    assert(n0 >= 0)
    assert(toOff0 <= to.length - n0)

    var toOff = toOff0
    var n = n0.toLong

    while (n > 0) {
      if (end == off) {
        val len = math.min(bufSize, n << 3).toInt
        in.readFully(buf, 0, len)
        end = len
        off = 0
      }
      val p = math.min(end - off, n << 3).toInt >>> 3
      assert(p > 0)
      Memory.memcpy(to, toOff, buf, off, p)
      toOff += p
      n -= p
      off += (p << 3)
    }
  }
}

final class DoubleOutputBuffer(out: OutputStream, bufSize: Int) extends Closeable {
  private val buf: Array[Byte] = new Array[Byte](bufSize)
  private var off: Int = 0

  def close(): Unit = {
    flush()
    out.close()
  }

  def flush(): Unit = {
    out.write(buf, 0, off)
  }

  def writeDoubles(from: Array[Double]): Unit = writeDoubles(from, 0, from.length)

  def writeDoubles(from: Array[Double], fromOff0: Int, n0: Int): Unit = {
    assert(n0 >= 0)
    assert(fromOff0 >= 0)
    assert(fromOff0 <= from.length - n0)
    var fromOff = fromOff0
    var n = n0.toLong

    while (off + (n << 3) > bufSize) {
      val p = (buf.length - off) >>> 3
      Memory.memcpy(buf, off, from, fromOff, p)
      off += (p << 3)
      fromOff += p
      n -= p
      out.write(buf, 0, off)
      off = 0
    }
    Memory.memcpy(buf, off, from, fromOff, n)
    off += (n.toInt << 3)
  }
}
