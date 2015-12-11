package org.broadinstitute.hail.io

import java.io.{FileOutputStream, DataOutputStream}
import org.apache.hadoop.fs.FSDataOutputStream

object ioUtils {
  //FIXME delete this stuff
  def writeIndexArray(arr: Array[Long], path: String) {
    val dos = new FSDataOutputStream(new FileOutputStream(path, false))
    dos.writeInt(arr.length)
    arr foreach {
      l => dos.writeLong(l)
    }
    dos.close()
  }

  def readIndexArray(path: String): Array[Long] = {
    val bbis = new BufferedBinaryReader(path)
    val variantsInArr = bbis.readIntBE()
    val indices = Array.ofDim[Long](variantsInArr)
    for (i <- indices.indices) {
      indices(i) = bbis.readLongBE()
    }
    indices
  }
}
