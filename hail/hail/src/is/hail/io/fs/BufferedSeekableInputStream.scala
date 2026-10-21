package is.hail.io.fs

import java.io.InputStream

class BufferedSeekableInputStream(raw: SeekableInputStream, capacity: Int)
    extends InputStream with Seekable {
  require(capacity > 0, s"capacity must be positive: $capacity")

  private[this] var buf: Array[Byte] = _
  // buf(bufPos until bufLimit) holds the bytes at file positions pos until pos + bufLimit - bufPos
  private[this] var bufPos: Int = 0
  private[this] var bufLimit: Int = 0
  private[this] var pos: Long = 0

  // Reads from raw until the buffer is full or raw is at EOF. Returns false iff nothing was read.
  private[this] def fill(): Boolean = {
    if (buf == null)
      buf = new Array[Byte](capacity)

    bufPos = 0
    bufLimit = 0
    var n = 0
    while (bufLimit < capacity && n != -1) {
      n = raw.read(buf, bufLimit, capacity - bufLimit)
      if (n > 0)
        bufLimit += n
    }
    bufLimit > 0
  }

  override def read(): Int = {
    if (bufPos == bufLimit && !fill())
      return -1

    val b = buf(bufPos) & 0xff
    bufPos += 1
    pos += 1
    b
  }

  override def read(bytes: Array[Byte], off: Int, len: Int): Int = {
    if (len == 0)
      return 0

    if (bufPos == bufLimit && len >= capacity) {
      /* A buffered copy would be no better than reading straight into the caller's array. Read all
       * of len: raw may return short reads, and a short read here would send the caller's next,
       * smaller request through fill(), reading ahead a full buffer it did not ask for. */
      bufPos = 0
      bufLimit = 0
      var nRead = 0
      var n = 0
      while (nRead < len && n != -1) {
        n = raw.read(bytes, off + nRead, len - nRead)
        if (n > 0)
          nRead += n
      }
      pos += nRead
      return if (nRead == 0) -1 else nRead
    }

    if (bufPos == bufLimit && !fill())
      return -1

    val n = math.min(len, bufLimit - bufPos)
    System.arraycopy(buf, bufPos, bytes, off, n)
    bufPos += n
    pos += n
    n
  }

  override def seek(newPos: Long): Unit = {
    val newBufPos = bufPos + (newPos - pos)
    if (newBufPos >= 0 && newBufPos <= bufLimit)
      bufPos = newBufPos.toInt
    else {
      bufPos = 0
      bufLimit = 0
      raw.seek(newPos)
    }
    pos = newPos
  }

  override def getPosition: Long = pos

  override def close(): Unit = raw.close()
}
