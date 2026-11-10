package is.hail.io.fs

import is.hail.TestUtils._
import is.hail.io.fs.BufferedSeekableInputStreamSuite.FakeRawStream

import java.io.InputStream

import org.junit.jupiter.api.Test

object BufferedSeekableInputStreamSuite {

  /** An in-memory raw stream that, like an unbuffered GCS ReadChannel, returns at most `maxChunk`
    * bytes per read, and counts the reads and seeks it is asked to perform.
    */
  private class FakeRawStream(data: Array[Byte], maxChunk: Int = 7)
      extends InputStream with Seekable {
    private[this] var pos: Long = 0
    var nReads: Int = 0
    var nSeeks: Int = 0
    var largestRead: Int = 0

    override def read(): Int = {
      val b = new Array[Byte](1)
      if (read(b, 0, 1) == -1) -1 else b(0) & 0xff
    }

    override def read(bytes: Array[Byte], off: Int, len: Int): Int = {
      nReads += 1
      largestRead = math.max(largestRead, len)
      if (pos >= data.length) return -1
      val n = math.min(math.min(len, maxChunk).toLong, data.length - pos).toInt
      System.arraycopy(data, pos.toInt, bytes, off, n)
      pos += n
      n
    }

    override def seek(newPos: Long): Unit = {
      nSeeks += 1
      pos = newPos
    }

    override def getPosition: Long = pos
  }
}

class BufferedSeekableInputStreamSuite {
  private[this] val data: Array[Byte] = Array.tabulate(1000)(i => (i % 251).toByte)

  @Test def testSequentialReadToEOF(): Unit = {
    val raw = new FakeRawStream(data)
    val is = new BufferedSeekableInputStream(raw, 64)

    assertEq(is.read(), 0)
    assertEq(is.getPosition, 1L)

    val rest = new Array[Byte](data.length - 1)
    var off = 0
    while (off < rest.length) {
      val n = is.read(rest, off, rest.length - off)
      assert(n > 0)
      off += n
    }
    assertEq(rest.toSeq, data.toSeq.drop(1))
    assertEq(is.getPosition, data.length.toLong)

    assertEq(is.read(), -1)
    assertEq(is.read(new Array[Byte](10), 0, 10), -1)
    assertEq(is.getPosition, data.length.toLong)
  }

  @Test def testSeekWithinBufferIsServedFromBuffer(): Unit = {
    val raw = new FakeRawStream(data)
    val is = new BufferedSeekableInputStream(raw, 64)

    assertEq(is.read(), 0)
    val readsAfterFill = raw.nReads

    is.seek(60)
    assertEq(is.getPosition, 60L)
    assertEq(is.read(), 60)

    is.seek(3)
    assertEq(is.getPosition, 3L)
    assertEq(is.read(), 3)

    is.seek(63)
    assertEq(is.read(), 63)

    assertEq(raw.nReads, readsAfterFill)
    assertEq(raw.nSeeks, 0)
  }

  @Test def testSeekOutsideBufferSeeksRaw(): Unit = {
    val raw = new FakeRawStream(data)
    val is = new BufferedSeekableInputStream(raw, 64)

    is.seek(500)
    assertEq(raw.nSeeks, 1)
    assertEq(is.getPosition, 500L)
    assertEq(is.read(), 500 % 251)

    is.seek(10)
    assertEq(raw.nSeeks, 2)
    assertEq(is.read(), 10)

    is.seek(10 + 64 + 1)
    assertEq(raw.nSeeks, 3)
    val bytes = new Array[Byte](4)
    assertEq(is.read(bytes, 0, 4), 4)
    assertEq(bytes.toSeq, data.slice(75, 79).toSeq)
    assertEq(is.getPosition, 79L)
  }

  @Test def testLargeReadBypassesEmptyBuffer(): Unit = {
    val raw = new FakeRawStream(data)
    val is = new BufferedSeekableInputStream(raw, 64)

    // raw returns short reads; the whole request is still read straight into the caller's array
    val bytes = new Array[Byte](300)
    assertEq(is.read(bytes, 0, 200), 200)
    assertEq(raw.largestRead, 200)
    assertEq(bytes.toSeq.take(200), data.toSeq.take(200))
    assertEq(is.getPosition, 200L)

    // nothing was buffered, so seeking back into what was just read goes to raw
    is.seek(100)
    assertEq(raw.nSeeks, 1)
    assertEq(is.read(), 100)
  }

  @Test def testLargeReadDrainsNonEmptyBufferFirst(): Unit = {
    val raw = new FakeRawStream(data, maxChunk = 1000)
    val is = new BufferedSeekableInputStream(raw, 64)

    assertEq(is.read(), 0)
    val readsAfterFill = raw.nReads

    val bytes = new Array[Byte](200)
    assertEq(is.read(bytes, 0, 200), 63)
    assertEq(bytes.toSeq.take(63), data.toSeq.slice(1, 64))
    assertEq(raw.nReads, readsAfterFill)
  }

  @Test def testSeekAfterEOF(): Unit = {
    val raw = new FakeRawStream(data)
    val is = new BufferedSeekableInputStream(raw, 64)

    is.seek(data.length - 2L)
    assertEq(is.read(), (data.length - 2) % 251)
    assertEq(is.read(), (data.length - 1) % 251)
    assertEq(is.read(), -1)
    assertEq(is.read(), -1)

    // back into the buffered window
    is.seek(data.length - 1L)
    assertEq(is.read(), (data.length - 1) % 251)
    assertEq(is.read(), -1)

    // outside the buffered window
    is.seek(0)
    assertEq(is.read(), 0)

    // past the end
    is.seek(data.length + 10L)
    assertEq(is.read(), -1)
    assertEq(is.read(new Array[Byte](100), 0, 100), -1)
  }
}
