package is.hail.misc

import java.io.InputStream

import is.hail.io.compress.BGzipInputStream
import org.apache.hadoop

object BGZipBlocks {
  //Print block starts of block gzip (bgz) file
  def apply(hadoopConf: hadoop.conf.Configuration, file: String) {
    var buf = new Array[Byte](64 * 1024)

    // position of `buf[0]' in input stream
    var bufPos = 0L

    var bufSize = 0
    var posInBuf = 0

    def fillBuf(is: InputStream) {
      val newSize = bufSize - posInBuf
      assert(newSize >= 0)

      System.arraycopy(buf, posInBuf, buf, 0, newSize)
      bufPos += posInBuf
      bufSize = newSize
      posInBuf = 0

      def f() {
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

    val hPath = new hadoop.fs.Path(file)
    val fs = hPath.getFileSystem(hadoopConf)

    // no decompression codec
    val is = fs.open(hPath)

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
