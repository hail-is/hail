package org.broadinstitute.hail.driver

import org.apache.hadoop
import java.io.InputStream
import org.broadinstitute.hail.io.compress._
import org.kohsuke.args4j.{Option => Args4jOption}

object BGZipBlocks extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: String = _

  }

  def newOptions = new Options

  def name = "bgzipblocks"

  def description = "Print block starts of block gzip (bgz) file"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
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

    val hConf = state.hadoopConf
    val hPath = new hadoop.fs.Path(options.input)
    val fs = hPath.getFileSystem(hConf)

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

    state
  }
}
