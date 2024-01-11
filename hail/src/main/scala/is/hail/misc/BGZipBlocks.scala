package is.hail.misc

import is.hail.io.compress.BGzipInputStream
import is.hail.io.fs.FS

import java.io.InputStream

object BGZipBlocks {
  // Print block starts of block gzip (bgz) file
  def apply(fs: FS, file: String): Unit = {
    val buf = new Array[Byte](64 * 1024)

    // position of 'buf[0]' in input stream
    var bufPos = 0L

    var bufSize = 0
    var posInBuf = 0

    def fillBuf(is: InputStream): Unit = {
      val newSize = bufSize - posInBuf
      assert(newSize >= 0)

      System.arraycopy(buf, posInBuf, buf, 0, newSize)
      bufPos += posInBuf
      bufSize = newSize
      posInBuf = 0

      def f(): Unit = {
        val needed = buf.length - bufSize
        if (needed > 0) {
          val result = is.read(buf, bufSize, needed)
          if (result > 0) {
            bufSize += result
            f()
          }
        }
      }

      f()
    }

    // no decompression codec
    val is = fs.open(file)

    fillBuf(is)

    while (bufSize > 0) {
      val h = new BGzipInputStream.BGzipHeader(buf, posInBuf, bufSize)
      println(bufPos)
      posInBuf += h.getBlockSize
      fillBuf(is)
    }

    is.close()
  }
}
