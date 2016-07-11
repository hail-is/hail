package org.broadinstitute.hail.utils

import java.io.IOException
import java.net.URI

import org.apache.hadoop

import scala.util.Random

object TempDir {
  def apply(tmpdir: String, hConf: hadoop.conf.Configuration): TempDir = {
    while (true) {
      try {
        val dirname = tmpdir + "/hail." + Random.alphanumeric.take(12).mkString

        hadoopMkdir(dirname, hConf)

        val fs = hadoopFS(tmpdir, hConf)
        fs.deleteOnExit(new hadoop.fs.Path(dirname))

        return new TempDir(dirname)
      } catch {
        case e: IOException =>
        // try again
      }
    }

    // can't happen
    null
  }
}

class TempDir(val dirname: String) {
  var counter: Int = 0

  def relFile(relPath: String) = dirname + "/" + relPath

  def relPath(relPath: String) =
    new URI(relFile(relPath)).getPath

  def createTempFile(prefix: String = "", extension: String = ""): String = {
    val i = counter
    counter += 1

    val sb = new StringBuilder
    sb.append(prefix)
    if (prefix != "")
      sb += '.'
    sb.append("%05d".format(i))
    sb.append(extension)

    relFile(sb.result())
  }
}
