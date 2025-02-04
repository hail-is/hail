package is.hail.io.compress

import is.hail.io.fs.FS

import java.nio.charset.StandardCharsets

final class BGzipLineReader(
  private val fs: FS,
  private val filePath: String,
) extends java.lang.AutoCloseable {
  private var is = new BGzipInputStream(fs.openNoCompression(filePath))

  // The line iterator buffer and associated state, we use this to avoid making
  // endless calls to read() (the no arg version returning int) on the input
  // stream, we start with 64k to make to make sure that we can always read
  // a block, and then grow it as necessary if a line is too large.
  //
  // The key invariant here is that bufferCursor should, except possibly in
  // readLine itself, be less than bufferLen. Should bufferCursor be greater
  // than or equal to bufferLen, we call refreshBuffer, which also saves the
  // current input stream virtual offset. If we uphold this invariant, then the
  // bufferCursor to bufferLen range within the buffer will always contain data
  // from the current block we are reading, meaning getVirtualOffset will
  // always return the same value as if we were calling is.read() to read the
  // lines and is.getVirtualOffset to update the current file pointer for every
  // call to next
  private var buffer = new Array[Byte](1 << 16)
  private var bufferCursor: Int = 0
  private var bufferLen: Int = 0
  private var bufferEOF: Boolean = false
  private var bufferPositionAtLastRead = 0L

  private var virtualFileOffsetAtLastRead = 0L

  def getVirtualOffset: Long =
    virtualFileOffsetAtLastRead + (bufferCursor - bufferPositionAtLastRead)

  def virtualSeek(l: Long): Unit = {
    is.virtualSeek(l)
    refreshBuffer(0)
    bufferCursor = 0
  }

  private def refreshBuffer(start: Int): Unit = {
    assert(start < buffer.length)
    bufferLen = start
    virtualFileOffsetAtLastRead = is.getVirtualOffset
    bufferPositionAtLastRead = start
    val bytesRead = is.read(buffer, start, buffer.length - start)
    if (bytesRead < 0)
      bufferEOF = true
    else {
      bufferLen = start + bytesRead
      bufferEOF = false
    }
    bufferCursor = start
  }

  private def decodeString(start: Int, end: Int): String = {
    var stop = end
    if (stop > start && buffer(stop) == '\r')
      stop -= 1
    val len = stop - start
    new String(buffer, start, len, StandardCharsets.UTF_8)
  }

  def readLine(): String = {
    assert(bufferCursor <= bufferLen)

    var start = bufferCursor

    while (true) {
      var str: String = null
      while (bufferCursor < bufferLen && buffer(bufferCursor) != '\n')
        bufferCursor += 1

      if (bufferCursor == bufferLen) { // no newline before end of buffer
        if (bufferEOF) {
          // `is` indicates end of file
          if (start == bufferCursor) {
            return null
          }
          str = decodeString(start, bufferCursor)
        } else if (bufferLen == buffer.length) {
          // line overflows buffer, need to increase buffer size
          val tmp = new Array[Byte](buffer.length * 2)
          System.arraycopy(buffer, 0, tmp, 0, buffer.length)
          buffer = tmp
          refreshBuffer(bufferLen)
        } else if (start > 0) {
          // line does not overflow buffer, but spans buffer.
          // Copy line left to the beginning of buffer and continue
          val nToCopy = bufferLen - start
          System.arraycopy(buffer, start, buffer, 0, nToCopy)
          start = 0
          refreshBuffer(nToCopy)
        } else {
          // line does not span buffer, but does not fill buffer either, read more
          refreshBuffer(bufferCursor)
        }
      } else { // found a newline
        str = decodeString(start, bufferCursor)
      }

      if (str != null) {
        bufferCursor += 1
        // we refresh here to make sure that the cursor is never pointing to
        // one past the end of a block, making it so getVirtualOffset will
        // never return the pointer pointing past the end of a block (the one
        // that overlaps with the start of the next block).
        if (bufferCursor >= bufferLen) {
          refreshBuffer(0)
        }
        return str
      }
    }
    throw new AssertionError()
  }

  override def close(): Unit =
    if (is != null) {
      is.close()
      is = null
    }
}
