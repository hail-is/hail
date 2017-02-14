package is.hail.utils

import java.io.IOException

import org.apache.hadoop

import scala.util.Random

object TempDir {
  def createTempDir(tmpdir: String, hConf: hadoop.conf.Configuration): String = {
    while (true) {
      try {
        val dir = tmpdir + "/hail." + Random.alphanumeric.take(12).mkString

        if (hConf.exists(dir)) {
          // try again
        } else {
          hConf.mkDir(dir)

          val fs = hConf.fileSystem(tmpdir)
          val qDir = fs.makeQualified(new hadoop.fs.Path(dir))
          fs.deleteOnExit(qDir)

          return qDir.toString
        }
      } catch {
        case e: IOException =>
        // try again
      }
    }

    // can't happen
    null
  }

  def apply(hConf: hadoop.conf.Configuration): TempDir =
    new TempDir(createTempDir("file:///tmp", hConf),
      createTempDir("/tmp", hConf))
}

class TempDir(localTempDir: String, tempDir: String) {
  var counter: Int = 0

  def createTempFile(tempDir: String, prefix: String, extension: String): String = {
    val i = counter
    counter += 1

    val sb = new StringBuilder
    sb.append(prefix)
    if (prefix.nonEmpty)
      sb += '.'
    sb.append("%05d".format(i))
    if (extension.nonEmpty)
      sb += '.'
    sb.append(extension)

    tempDir + "/" + sb.result()
  }

  def createTempFile(prefix: String = "", extension: String = ""): String =
    createTempFile(tempDir, prefix, extension)

  def createLocalTempFile(prefix: String = "", extension: String = ""): String =
    createTempFile(localTempDir, prefix, extension)
}
