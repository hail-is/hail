package org.broadinstitute.hail.io

import java.io.{FileOutputStream, DataOutputStream}

object ioUtils {

  def writeIndexArray(arr: Array[Long], path: String) {
    val dos = new DataOutputStream(new FileOutputStream(path, false))  // false means overwrite existing
    dos.writeInt(arr.length)
    arr foreach {
      l => dos.writeLong(l)
    }
    dos.close()
  }

  def readIndexArray(path: String): Array[Long] = {
    val bbis = new BetterBufferedInputStream(path)
    val variantsInArr = bbis.readIntBE()
    val indices = Array.ofDim[Long](variantsInArr)
    for (i <- indices.indices) {
      indices(i) = bbis.readLongBE()
    }
    indices
  }
}
